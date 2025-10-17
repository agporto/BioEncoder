#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% imports


import argparse
import logging
import os
import time
import shutil
import sys 
from rich.pretty import pretty_repr

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage

from bioencoder import config, utils

#%% function

def train(
    config_path,
    dry_run=False,
    overwrite=False,
    **kwargs,
):
    """
    Trains the BioEncoder model based on the provided configuration settings in the yaml files.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file that specifies detailed training and optimizer parameters.
        This file controls various aspects of the training process including but not limited to model architecture,
        optimizer settings, scheduler details, and data augmentation strategies.
    overwrite : bool, optional
        If True, existing directories for logs, runs, and weights will be removed and recreated, allowing for a clean training start.
        If False, the training process will append to existing directories and files, which could lead to mixed results if not managed properly.
        Default is False.

    Raises
    ------
    FileNotFoundError
        If the configuration file specified by `config_path` does not exist.
    AssertionError
        If certain conditions in the configuration (like minimum image count per class) are not met.
    ValueError
        If incompatible or inconsistent parameters are detected during the setup or training processes.

    Notes
    -----
    There are two separate files for the first and second stage - make sure you set them up appropriately. E.g., for stage two,
    specify the number of classes, and a different learning rate.

    Examples
    --------
    To start a new training session with overwriting previous outputs:
        bioencoder.train("/path/to/config.yaml", overwrite=True)

    """
    
    ## load bioencoer config
    root_dir = config.root_dir
    run_name = config.run_name
    
    ## load config
    hyperparams = utils.load_yaml(config_path)

    ## parse config
    backbone = hyperparams["model"]["backbone"]
    amp = hyperparams["train"]["amp"]
    ema = hyperparams["train"]["ema"]
    ema_decay_per_epoch = hyperparams["train"]["ema_decay_per_epoch"]
    n_epochs = hyperparams["train"]["n_epochs"]
    target_metric = hyperparams["train"]["target_metric"]
    stage = hyperparams["train"]["stage"]
    optimizer_params = hyperparams["optimizer"]
    scheduler_params = hyperparams["scheduler"]
    criterion_params = hyperparams["criterion"]
    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        "valid_batch_size": hyperparams["dataloaders"]["valid_batch_size"],
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]
    aug_sample = hyperparams["augmentations"].get("sample_save", False)
    aug_sample_n = hyperparams["augmentations"].get("sample_n", 5)
    aug_sample_seed = hyperparams["augmentations"].get("sample_seed", 42)

    ## manage directories and paths
    data_dir = os.path.join(root_dir, "data", run_name)
    log_dir = os.path.join(root_dir, "logs", run_name, stage)
    run_dir = os.path.join(root_dir, "runs", run_name, stage)
    weights_dir = os.path.join(root_dir, "weights", run_name, stage)
    # Create output directories (rank 0 only to avoid races)
    if os.path.exists(log_dir) and overwrite and os.path.isdir(log_dir):
        pass
    if is_main_process:
        for directory in [log_dir, run_dir, weights_dir]:
            if os.path.exists(directory) and overwrite==True:
                print(f"removing {directory} (overwrite=True)")
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        
    ## collect information on data
    train_dir, val_dir = os.path.join(data_dir, "train"), os.path.join(data_dir, "val")
    class_names, data_stats = os.listdir(train_dir),{"data_dir": data_dir}
    data_stats["train"], data_stats["val"] = {},{}
    for class_name in class_names:
        data_stats["train"][class_name] = len(os.listdir(os.path.join(train_dir, class_name)))
        data_stats["val"][class_name] = len(os.listdir(os.path.join(val_dir, class_name)))

    # --- distributed init ---
    def init_distributed():
        distributed = False
        rank = 0
        world_size = 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if os.environ.get("WORLD_SIZE") is not None and int(os.environ["WORLD_SIZE"]) > 1:
            distributed = True
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ["WORLD_SIZE"])
        if distributed:
            backend = "nccl"
            # Fallback for platforms without NCCL (e.g., Windows)
            try:
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
            except Exception:
                backend = "gloo"
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
        else:
            # single GPU or CPU fallback
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        return distributed, rank, world_size, local_rank

    distributed, rank, world_size, local_rank = init_distributed()

    is_main_process = (rank == 0)

    ## set up logging and tensorboard writer (rank 0 only)
    writer = SummaryWriter(run_dir) if is_main_process else None
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    log_file_path = os.path.join(log_dir, f"{run_name}_{stage}.log")

    if is_main_process:
        ## logging: stdout handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_formatter = logging.Formatter('%(asctime)s: %(message)s', "%H:%M:%S")
        stdout_handler.setFormatter(stdout_formatter)
        logger.addHandler(stdout_handler)

        ## logging: logfile handler
        if os.path.isfile(log_file_path):
            os.remove(log_file_path)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    else:
        # Avoid warnings for no handlers on non-main ranks
        logger.addHandler(logging.NullHandler())
    
    ## manage second stage
    if stage == "second":

        ## number of classes
        num_classes = hyperparams["model"]["num_classes"]
        
        ## add learning rate
        if not "params" in optimizer_params:
            optimizer_params["params"] = {}
        if kwargs.get("lr"):
            optimizer_params["params"]["lr"] = kwargs.get("lr")
        if not "lr" in optimizer_params["params"].keys():
            if "second_lr" in config.__dict__.keys():
                optimizer_params["params"] = {"lr": float(config.second_lr)}
                logger.info(f"Using LR value from global bioencoder config: {config.second_lr}")
        else:
            lr = optimizer_params["params"]["lr"]
            logger.info(f"Using LR value from local bioencoder config: {lr}")
            
        assert "lr" in optimizer_params["params"], "no learning rate specified"
        
        ## fetch checkpoints from first stage
        ckpt_pretrained = os.path.join(root_dir, "weights", run_name, 'first', "swa") 
    else: 
        ckpt_pretrained = None   
        num_classes = None

    ## add hyperparams to log
    logger.info(utils.pprint_fill_hbar(f"Training {stage} stage ", symbol="#"))
    logger.info(f"Dataset:\n{pretty_repr(data_stats)}")
    logger.info(f"Hyperparameters:\n{pretty_repr(hyperparams)}")
    
    ## scaler
    scaler = torch.cuda.amp.GradScaler()
    if not amp:
        scaler = None
        
    ## set seed for entire pipeline
    utils.set_seed()

    # create model, loaders, optimizer, etc
    transforms = utils.build_transforms(hyperparams)    
    loaders = utils.build_loaders(
        data_dir, 
        transforms, 
        batch_sizes, 
        num_workers, 
        second_stage=(stage == "second"), 
        is_supcon=(criterion_params["name"] == "SupCon"),
        distributed=distributed,
    )
    model = utils.build_model(
        backbone,
        second_stage=(stage == "second"),
        num_classes=num_classes,
        ckpt_pretrained=ckpt_pretrained,
        cuda_device=local_rank,
    ).cuda(local_rank)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    ## save a sample of augmented images
    if aug_sample and is_main_process:
        utils.save_augmented_sample(data_dir, transforms["train_transforms"], aug_sample_n, seed=aug_sample_seed)
        logger.info(f"Saving augmentation samples: {aug_sample_n} per class to data/{run_name}/aug_sample")

    logger.info(f"Using backbone: {backbone}")
    
    ## configure GPU 
    if is_main_process:
        assert torch.cuda.device_count() > 0, "No GPUs detected on this System (check your CUDA setup) - aborting."
        if distributed:
            logger.info(f"Distributed training enabled: world_size={world_size}, backend={dist.get_backend() if dist.is_initialized() else 'N/A'}")
            logger.info(f"Running on device {local_rank}: {torch.cuda.get_device_name(local_rank)}")
        else:
            if torch.cuda.device_count() == 1:
                logger.info(f"Found one GPU: {torch.cuda.get_device_name(0)} (device {torch.cuda.current_device()})")
            else:
                logger.info(f"Found {torch.cuda.device_count()} GPUs; using single-process training on GPU {torch.cuda.current_device()}")
        
    ## configure optimizer
    optim = utils.build_optim(
        model, optimizer_params, scheduler_params, criterion_params
    )
    criterion, optimizer, scheduler, loss_optimizer = (
        optim["criterion"],
        optim["optimizer"],
        optim["scheduler"],
        optim["loss_optimizer"],
    )
    if ema and is_main_process:
        iters = len(loaders["train_loader"])
        ema_decay = ema_decay_per_epoch ** (1 / iters)
        # EMA only on main process
        ema = ExponentialMovingAverage((model.module if isinstance(model, DDP) else model).parameters(), decay=ema_decay)
    else:
        ema = None

    if loss_optimizer is not None and stage == 'second':
        raise ValueError('Loss optimizers should only be present for stage 1 training. Check your config file.') 
            
    # epoch loop
    metric_best = 0
    if not dry_run:
        for epoch in range(n_epochs):
            logger.info(utils.pprint_fill_hbar(f"START - Epoch {epoch}"))
            # Distributed sampler epoch shuffling
            if distributed and 'train_sampler' in loaders:
                loaders['train_sampler'].set_epoch(epoch)
            if distributed and 'train_supcon_sampler' in loaders:
                loaders['train_supcon_sampler'].set_epoch(epoch)
            start_training_time = time.time()
            if stage == "first":
                train_metrics = utils.train_epoch_constructive(
                    loaders["train_supcon_loader"], 
                    model, 
                    criterion, 
                    optimizer, 
                    scaler, 
                    ema, 
                    loss_optimizer
                )
            else:
                train_metrics = utils.train_epoch_ce(
                    loaders["train_loader"],
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    ema,
                )
            end_training_time = time.time()
    
            if ema and is_main_process:
                m_ref = model.module if isinstance(model, DDP) else model
                copy_of_model_parameters = utils.copy_parameters_from_model(m_ref)
                ema.copy_to(m_ref.parameters())
    
            start_validation_time = time.time()
            if distributed and dist.is_initialized():
                dist.barrier()
    
            if is_main_process:
                if stage == "first":
                    # use non-sampled train_eval_loader for reference embeddings
                    valid_metrics_projection_head = utils.validation_constructive(
                        loaders["valid_loader"], loaders.get("train_eval_loader", loaders["train_loader"]), model, scaler
                    )
                    model_ref = model.module if isinstance(model, DDP) else model
                    model_ref.use_projection_head(False)
                    valid_metrics_encoder = utils.validation_constructive(
                        loaders["valid_loader"], loaders.get("train_eval_loader", loaders["train_loader"]), model, scaler
                    )
                    model_ref.use_projection_head(True)

                    ## epoch summary
                    message = "Summary epoch {}:\ntrain time {:.2f}\nvalid time {:.2f}\ntrain loss {:.2f}\nvalid acc projection head {}\nvalid acc encoder {}".format(
                        epoch,
                        end_training_time - start_training_time,
                        time.time() - start_validation_time,
                        train_metrics["loss"],
                        pretty_repr(valid_metrics_projection_head),
                        pretty_repr(valid_metrics_encoder),
                    )
                    logger.info("\n".join(line if i == 0 else "    " + line for i, line in enumerate(message.split("\n"))))
                    valid_metrics = valid_metrics_projection_head
                else:
                    valid_metrics = utils.validation_ce(
                        model, criterion, loaders["valid_loader"], scaler
                    )
                    ## epoch summary
                    message =  "Summary epoch {}:\ntrain time {:.2f}\nvalid time {:.2f}\ntrain loss {:.2f}\nvalid acc dict {}".format(
                            epoch,
                            end_training_time - start_training_time,
                            time.time() - start_validation_time,
                            train_metrics["loss"],
                            pretty_repr(valid_metrics),
                    )
                    logger.info("\n".join(line if i == 0 else "    " + line for i, line in enumerate(message.split("\n"))))
    
            # write train and valid metrics to the logs
            if is_main_process and writer is not None:
                utils.add_to_tensorboard_logs(
                    writer, train_metrics["loss"], "Loss/train", epoch
                )
                if 'valid_metrics' in locals():
                    for valid_metric in valid_metrics:
                        try:
                            utils.add_to_tensorboard_logs(
                                writer,
                                valid_metrics[valid_metric],
                                "{}/validation".format(valid_metric),
                                epoch,
                            )
                        except AssertionError:
                            pass
    
            # check if the best value of metric changed. If so -> save the model
            if is_main_process:
                if (
                    valid_metrics[target_metric] > metric_best*0.99
                ):  # > 0 if wanting to save all models 
                    logger.info(
                        "{} increased ({:.6f} --> {:.6f}).  Saving model ...".format(
                            target_metric, metric_best, valid_metrics[target_metric]
                        )
                    )

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        os.path.join(weights_dir, f"epoch{epoch}"),
                    )
                    metric_best = valid_metrics[target_metric]
    
            # if ema is used, go back to regular weights without ema
            if ema and is_main_process:
                m_ref = model.module if isinstance(model, DDP) else model
                utils.copy_parameters_to_model(copy_of_model_parameters, m_ref)
    
            scheduler.step()
            logger.info(utils.pprint_fill_hbar(f"END - Epoch {epoch}"))
    else:
        if is_main_process:
            logger.info(utils.pprint_fill_hbar("DRY-RUN ONLY - NO TRAINING"))
    if writer is not None:
        writer.close()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    logging.shutdown()


def cli():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",type=str, help="Path to the YAML configuration file that specifies detailed training and optimizer parameters.")
    parser.add_argument("--dry-run", action='store_true', help="Run without starting the training to inspect config and augmentations.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files without asking.")
    args = parser.parse_args()
    
    train_cli = utils.restore_config(train)
    train_cli(args.config_path, overwrite=args.overwrite, dry_run=args.dry_run)

if __name__ == "__main__":
    
    cli()
