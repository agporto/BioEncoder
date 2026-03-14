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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

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
    root_dir = config.root_dir
    run_name = config.run_name
    hyperparams = utils.load_yaml(config_path)

    backbone = hyperparams["model"]["backbone"]
    amp = hyperparams["train"]["amp"]
    ema = hyperparams["train"]["ema"]
    ema_decay_per_epoch = hyperparams["train"]["ema_decay_per_epoch"]
    progress_bar = hyperparams["train"].get("progress_bar", True)
    n_epochs = hyperparams["train"]["n_epochs"]
    target_metric = hyperparams["train"]["target_metric"]
    min_improvement = hyperparams["train"].get("min_improvement", 0.01)
    target_metric_mode = hyperparams["train"].get("target_metric_mode", "auto")
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
    dist_config = hyperparams.get("distributed", {})
    distributed_enabled = kwargs.get("distributed", dist_config.get("enabled", False))
    distributed_backend = kwargs.get("backend", dist_config.get("backend", "nccl"))
    find_unused_parameters = dist_config.get("find_unused_parameters", False)
    sync_bn = dist_config.get("sync_bn", False)
    grad_accum_steps = dist_config.get("grad_accum_steps", 1)
    seed = dist_config.get("seed", 42)
    env_rank = int(os.environ.get("RANK", "0"))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if distributed_enabled:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        utils.init_distributed(backend=distributed_backend, local_rank=local_rank)
        env_rank = utils.get_rank()
        env_world_size = utils.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(root_dir, "data", run_name)
    log_dir = os.path.join(root_dir, "logs", run_name, stage)
    run_dir = os.path.join(root_dir, "runs", run_name, stage)
    weights_dir = os.path.join(root_dir, "weights", run_name, stage)

    setup_token = os.path.join(root_dir, f".setup_{run_name}_{stage}.done")

    if distributed_enabled:
        if local_rank == 0:
            if os.path.exists(setup_token):
                os.remove(setup_token)
            for directory in [log_dir, run_dir, weights_dir]:
                if os.path.exists(directory) and overwrite:
                    shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)
            with open(setup_token, "w") as f:
                f.write("ok\n")
        else:
            t0 = time.time()
            while not os.path.exists(setup_token):
                if time.time() - t0 > 300:
                    raise TimeoutError(f"Timed out waiting for setup token: {setup_token}")
                time.sleep(0.1)
            for directory in [log_dir, run_dir, weights_dir]:
                os.makedirs(directory, exist_ok=True)
    else:
        for directory in [log_dir, run_dir, weights_dir]:
            if os.path.exists(directory) and overwrite:
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    class_names = sorted(
        [
            class_name
            for class_name in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, class_name))
        ]
    )

    data_stats = {"data_dir": data_dir, "train": {}, "val": {}}
    for class_name in class_names:
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        data_stats["train"][class_name] = len(
            [f for f in os.listdir(train_class_dir) if os.path.isfile(os.path.join(train_class_dir, f))]
        )
        data_stats["val"][class_name] = len(
            [f for f in os.listdir(val_class_dir) if os.path.isfile(os.path.join(val_class_dir, f))]
        )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    log_file_path = os.path.join(log_dir, f"{run_name}_{stage}.log")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(stdout_handler)

    if (not distributed_enabled) or local_rank == 0:
        if os.path.isfile(log_file_path):
            os.remove(log_file_path)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    logger.info(utils.pprint_fill_hbar(f"Training {stage} stage ", symbol="#"))
    logger.info(f"Dataset:\n{pretty_repr(data_stats)}")
    logger.info(f"Hyperparameters:\n{pretty_repr(hyperparams)}")

    if stage == "second":
        num_classes = hyperparams["model"]["num_classes"]
        if "params" not in optimizer_params:
            optimizer_params["params"] = {}
        if kwargs.get("lr") is not None:
            optimizer_params["params"]["lr"] = kwargs.get("lr")
        if "lr" not in optimizer_params["params"]:
            if "second_lr" in config.__dict__:
                optimizer_params["params"]["lr"] = float(config.second_lr)
                logger.info(f"Using LR value from global bioencoder config: {config.second_lr}")
            elif "lr" in config.__dict__:
                optimizer_params["params"]["lr"] = float(config.lr)
                logger.info(f"Using LR value from global bioencoder config: {config.lr}")
        else:
            logger.info(f"Using LR value from local bioencoder config: {optimizer_params['params']['lr']}")
        assert "lr" in optimizer_params["params"], "no learning rate specified"
        ckpt_pretrained = os.path.join(root_dir, "weights", run_name, "first", "swa")
    else:
        num_classes = None
        ckpt_pretrained = None

    transforms = utils.build_transforms(hyperparams)

    loaders = utils.build_loaders(
        data_dir=data_dir,
        transforms=transforms,
        batch_sizes=batch_sizes,
        num_workers=num_workers,
        second_stage=(stage == "second"),
        is_supcon=(criterion_params["name"] == "SupCon"),
        distributed=distributed_enabled,
        rank=env_rank,
        world_size=env_world_size,
    )

    train_sampler = loaders["train_loader"].sampler if distributed_enabled else None
    train_supcon_sampler = (
        loaders["train_supcon_loader"].sampler
        if distributed_enabled and "train_supcon_loader" in loaders
        else None
    )

    if distributed_enabled:
        logger.info(
            f"DDP initialized (rank={utils.get_rank()}/{utils.get_world_size()}, local_rank={local_rank}, backend={distributed_backend})"
        )
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    utils.set_seed(seed, rank_offset=(local_rank if distributed_enabled else 0))

    scaler = torch.amp.GradScaler("cuda") if amp else None

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logger.info("No GPU found. Using CPU.")
    elif gpu_count == 1:
        logger.info(f"Found one GPU: {torch.cuda.get_device_name(0)} (device {torch.cuda.current_device()})")
    else:
        if distributed_enabled:
            logger.info(f"Found {gpu_count} GPUs and using DDP across {utils.get_world_size()} ranks")
        else:
            logger.info(f"Found {gpu_count} GPUs, but distributed mode is disabled.")
            logger.info(f"Using GPU {torch.cuda.get_device_name(0)} (device {torch.cuda.current_device()})")

    writer = SummaryWriter(run_dir) if ((not distributed_enabled) or utils.is_main_process()) else None

    model = utils.build_model(
        backbone,
        second_stage=(stage == "second"),
        num_classes=num_classes,
        ckpt_pretrained=ckpt_pretrained,
        cuda_device=device,
    ).to(device)
    model = model.to(device)

    if distributed_enabled and sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if distributed_enabled:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )

    if aug_sample and ((not distributed_enabled) or utils.is_main_process()):
        utils.save_augmented_sample(data_dir, transforms["train_transforms"], aug_sample_n, seed=aug_sample_seed)
        logger.info(f"Saving augmentation samples: {aug_sample_n} per class to data/{run_name}/aug_sample")

    logger.info(f"Using backbone: {backbone}")

    optim = utils.build_optim(model, optimizer_params, scheduler_params, criterion_params)

    criterion = optim["criterion"]
    optimizer = optim["optimizer"]
    scheduler = optim["scheduler"]
    loss_optimizer = optim["loss_optimizer"]

    scheduler_step_per_batch = isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)
    scheduler_requires_metric = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    if ema:
        active_train_loader = loaders["train_supcon_loader"] if stage == "first" else loaders["train_loader"]
        iters = len(active_train_loader)
        ema_decay = ema_decay_per_epoch ** (1 / iters)
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    if loss_optimizer is not None and stage == "second":
        raise ValueError("Loss optimizers should only be present for stage 1 training. Check your config file.")

    if target_metric_mode not in {"auto", "max", "min"}:
        raise ValueError("target_metric_mode must be one of: auto, max, min")
    if target_metric_mode == "auto":
        mode = "min" if "loss" in str(target_metric).lower() else "max"
    else:
        mode = target_metric_mode

    metric_best = float("inf") if mode == "min" else float("-inf")

    try:
        if not dry_run:
            for epoch in range(n_epochs):
                if distributed_enabled:
                    if train_sampler is not None:
                        train_sampler.set_epoch(epoch)
                    if train_supcon_sampler is not None:
                        train_supcon_sampler.set_epoch(epoch)

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
                        device=device,
                        grad_accum_steps=grad_accum_steps,
                        progress_bar=progress_bar,
                        epoch=epoch,
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
                        device=device,
                        grad_accum_steps=grad_accum_steps,
                        progress_bar=progress_bar,
                        epoch=epoch,
                    )

                end_training_time = time.time()

                if ema:
                    copy_of_model_parameters = utils.copy_parameters_from_model(model)
                    ema.copy_to(model.parameters())

                start_validation_time = time.time()

                if stage == "first":
                    valid_metrics_projection_head = utils.validation_constructive(
                        loaders["valid_loader"],
                        loaders["train_loader"],
                        model,
                        device,
                        scaler,
                        progress_bar=progress_bar,
                        epoch=epoch,
                        split_name="projection",
                    )
                    model_ref = model.module if isinstance(model, DDP) else model
                    model_ref.use_projection_head(False)
                    valid_metrics_encoder = utils.validation_constructive(
                        loaders["valid_loader"],
                        loaders["train_loader"],
                        model,
                        device,
                        scaler,
                        progress_bar=progress_bar,
                        epoch=epoch,
                        split_name="encoder",
                    )
                    model_ref.use_projection_head(True)

                    message = (
                        "Summary epoch {}:\ntrain time {:.2f}\nvalid time {:.2f}\ntrain loss {:.2f}\n"
                        "valid acc projection head {}\nvalid acc encoder {}"
                    ).format(
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
                        model,
                        criterion,
                        loaders["valid_loader"],
                        device,
                        scaler,
                        progress_bar=progress_bar,
                        epoch=epoch,
                    )
                    message = (
                        "Summary epoch {}:\ntrain time {:.2f}\nvalid time {:.2f}\ntrain loss {:.2f}\nvalid acc dict {}"
                    ).format(
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

                if (not distributed_enabled) or utils.is_main_process():
                    utils.add_to_tensorboard_logs(writer, train_metrics["loss"], "Loss/train", epoch)
                    for valid_metric in valid_metrics:
                        try:
                            utils.add_to_tensorboard_logs(
                                writer,
                                valid_metrics[valid_metric],
                                f"{valid_metric}/validation",
                                epoch,
                            )
                        except AssertionError:
                            pass

                current_metric = valid_metrics[target_metric]
                if mode == "max":
                    improved = current_metric > metric_best * (1 + min_improvement)
                else:
                    improved = current_metric < metric_best * (1 - min_improvement)
                if improved:
                    logger.info(
                        "{} improved by ≥{:.2%} ({:.6f} --> {:.6f}). Saving model ...".format(
                            target_metric, min_improvement, metric_best, current_metric
                        )
                    )
                    if (not distributed_enabled) or utils.is_main_process():
                        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model_state,
                                "optimizer_state_dict": optimizer.state_dict(),
                            },
                            os.path.join(weights_dir, f"epoch{epoch}"),
                        )
                    metric_best = current_metric
                else:
                    logger.info(
                        f"Metric {target_metric} did not improve by ≥{min_improvement:.2%} "
                        f"(best: {metric_best:.6f}, current: {current_metric:.6f})"
                    )

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
    finally:
        if writer is not None:
            writer.close()
        if distributed_enabled and utils.is_distributed():
            utils.teardown_distributed()
        if distributed_enabled and local_rank == 0 and os.path.exists(setup_token):
            os.remove(setup_token)
        logging.shutdown()

def cli():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",type=str, required=True, help="Path to the YAML configuration file that specifies detailed training and optimizer parameters.")
    parser.add_argument("--dry-run", action='store_true', help="Run without starting the training to inspect config and augmentations.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files without asking.")
    parser.add_argument("--distributed", action='store_true', help="Enable Distributed Data Parallel training.")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend.")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None, help="Local rank set by torchrun.")
    args = parser.parse_args()
    
    train_cli = utils.restore_config(train)
    train_cli(
        args.config_path,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        distributed=args.distributed,
        backend=args.backend,
        local_rank=args.local_rank,
    )

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cli()
