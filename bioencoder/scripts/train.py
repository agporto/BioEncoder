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
    min_improvement = hyperparams["train"].get("min_improvement", 0.01)
    stage = hyperparams["train"]["stage"]
    optimizer_params = hyperparams["optimizer"]
    scheduler_params = hyperparams.get("scheduler", None)
    criterion_params = hyperparams["criterion"]
    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        "valid_batch_size": hyperparams["dataloaders"]["valid_batch_size"],
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]
    aug_config = hyperparams.get("augmentations", {})
    aug_sample = aug_config.get("sample_save", False)
    aug_sample_n = aug_config.get("sample_n", 5)
    aug_sample_seed = aug_config.get("sample_seed", 42)

    ## manage directories and paths
    data_dir = os.path.join(root_dir, "data", run_name)
    log_dir = os.path.join(root_dir, "logs", run_name, stage)
    run_dir = os.path.join(root_dir, "runs", run_name, stage)
    weights_dir = os.path.join(root_dir, "weights", run_name, stage)
    for directory in [log_dir, run_dir, weights_dir]:
        if os.path.exists(directory) and overwrite==True:
            print(f"removing {directory} (overwrite=True)")
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
        
    ## collect information on data
    train_dir, val_dir = os.path.join(data_dir, "train"), os.path.join(data_dir, "val")
    class_names = sorted(
        [
            class_name
            for class_name in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, class_name))
        ]
    )
    data_stats = {"data_dir": data_dir}
    data_stats["train"], data_stats["val"] = {},{}
    for class_name in class_names:
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        data_stats["train"][class_name] = len(
            [
                file_name
                for file_name in os.listdir(train_class_dir)
                if os.path.isfile(os.path.join(train_class_dir, file_name))
            ]
        )
        data_stats["val"][class_name] = len(
            [
                file_name
                for file_name in os.listdir(val_class_dir)
                if os.path.isfile(os.path.join(val_class_dir, file_name))
            ]
        )

    ## set up logging and tensorboard writer
    writer = SummaryWriter(run_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    log_file_path = os.path.join(log_dir, f"{run_name}_{stage}.log")
    
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
                optimizer_params["params"]["lr"] = float(config.second_lr)
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
    scaler = torch.amp.GradScaler("cuda")
    if not amp:
        scaler = None
        
    ## set seed for entire pipeline
    utils.set_seed()

    ## configure GPU before moving model to CUDA
    assert torch.cuda.device_count() > 0, "No GPUs detected on this System (check your CUDA setup) - aborting."
    if torch.cuda.device_count() == 1:
        logger.info(f"Found one GPU: {torch.cuda.get_device_name(0)} (device {torch.cuda.current_device()})")
    else:
        logger.info(f"Found {torch.cuda.device_count()} GPUs, but unfortunately multi-GPU use isn't implemented yet.")
        logger.info(f"Using GPU {torch.cuda.get_device_name(0)} (device {torch.cuda.current_device()})")

    # create model, loaders, optimizer, etc
    transforms = utils.build_transforms(hyperparams)    
    loaders = utils.build_loaders(
        data_dir, 
        transforms, 
        batch_sizes, 
        num_workers, 
        second_stage=(stage == "second"), 
        is_supcon=(criterion_params["name"] == "SupCon"),
    )
    model = utils.build_model(
        backbone,
        second_stage=(stage == "second"),
        num_classes=num_classes,
        ckpt_pretrained=ckpt_pretrained,
    ).cuda()
    
    ## save a sample of augmented images
    if aug_sample:
        utils.save_augmented_sample(data_dir, transforms["train_transforms"], aug_sample_n, seed=aug_sample_seed)
        logger.info(f"Saving augmentation samples: {aug_sample_n} per class to data/{run_name}/aug_sample")

    logger.info(f"Using backbone: {backbone}")

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
    scheduler_step_per_batch = isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)
    scheduler_requires_metric = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    if ema:
        active_train_loader = loaders["train_supcon_loader"] if stage == "first" else loaders["train_loader"]
        iters = len(active_train_loader)
        ema_decay = ema_decay_per_epoch ** (1 / iters)
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    if loss_optimizer is not None and stage == 'second':
        raise ValueError('Loss optimizers should only be present for stage 1 training. Check your config file.') 
            
    # epoch loop
    metric_best = 0
    if not dry_run:
        for epoch in range(n_epochs):
            logger.info(utils.pprint_fill_hbar(f"START - Epoch {epoch}"))
            start_training_time = time.time()
            if stage == "first":
                train_metrics = utils.train_epoch_constructive(
                    loaders["train_supcon_loader"], 
                    model, 
                    criterion, 
                    optimizer, 
                    scaler, 
                    ema, 
                    loss_optimizer,
                    scheduler=scheduler,
                    scheduler_step_per_batch=scheduler_step_per_batch,
                )
            else:
                train_metrics = utils.train_epoch_ce(
                    loaders["train_loader"],
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    ema,
                    scheduler=scheduler,
                    scheduler_step_per_batch=scheduler_step_per_batch,
                )
            end_training_time = time.time()
    
            if ema:
                copy_of_model_parameters = utils.copy_parameters_from_model(model)
                ema.copy_to(model.parameters())
    
            start_validation_time = time.time()
    
            if stage == "first":
                valid_metrics_projection_head = utils.validation_constructive(
                    loaders["valid_loader"], loaders["train_loader"], model, scaler
                )
                
                ## check for GPU parallelization
                #model_copy = model.module if isinstance(model, torch.nn.DataParallel) else model
                
                #model_copy.use_projection_head(False)
                model.use_projection_head(False)
                valid_metrics_encoder = utils.validation_constructive(
                    loaders["valid_loader"], loaders["train_loader"], model, scaler
                )
                model.use_projection_head(True)
                #model_copy.use_projection_head(True)    parser.add_argument("--dry_run", action='store_true', help="Run without making any changes.")

                
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

            if target_metric not in valid_metrics:
                raise ValueError(
                    f"target_metric='{target_metric}' not found in validation metrics. "
                    f"Available metrics: {list(valid_metrics.keys())}"
                )
    
            # write train and valid metrics to the logs
            utils.add_to_tensorboard_logs(
                writer, train_metrics["loss"], "Loss/train", epoch
            )
            for valid_metric in valid_metrics:
                try:
                    utils.add_to_tensorboard_logs(
                        writer,
                        valid_metrics[valid_metric],
                        "{}/validation".format(valid_metric),
                        epoch,
                    )
                except AssertionError:
                    # in case valid metric is a listhyperparams
                    pass
    
            # check if the best value of metric changed. If so -> save the model
            current_metric = valid_metrics[target_metric]
            if metric_best == 0 or current_metric > metric_best * (1 + min_improvement):
                logger.info(
                    "{} improved by ≥{:.2%} ({:.6f} --> {:.6f}). Saving model ...".format(
                        target_metric, min_improvement, metric_best, current_metric
                    )
                )
   
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(weights_dir, f"epoch{epoch}"),
                )
                metric_best = current_metric
            else:
                logger.info(f"Metric {target_metric} did not improve by ≥{min_improvement:.2%} (best: {metric_best:.6f}, current: {current_metric:.6f})")

            # if ema is used, go back to regular weights without ema
            if ema:
                utils.copy_parameters_to_model(copy_of_model_parameters, model)
    
            if scheduler is not None:
                if scheduler_requires_metric:
                    scheduler.step(valid_metrics[target_metric])
                elif not scheduler_step_per_batch:
                    scheduler.step()
            logger.info(utils.pprint_fill_hbar(f"END - Epoch {epoch}"))
    else:
        logger.info(utils.pprint_fill_hbar("DRY-RUN ONLY - NO TRAINING"))
    writer.close()
    logging.shutdown()


def cli():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",type=str, required=True, help="Path to the YAML configuration file that specifies detailed training and optimizer parameters.")
    parser.add_argument("--dry-run", action='store_true', help="Run without starting the training to inspect config and augmentations.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files without asking.")
    args = parser.parse_args()
    
    train_cli = utils.restore_config(train)
    train_cli(args.config_path, overwrite=args.overwrite, dry_run=args.dry_run)

if __name__ == "__main__":
    
    cli()
