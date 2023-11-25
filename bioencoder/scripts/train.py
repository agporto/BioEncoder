import argparse
import logging
import os
import time
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
import torch

from bioencoder import utils
from bioencoder import config



def train(
    config_path,
):
    
    ## load config
    hyperparams = utils.load_config(config_path)

    ## parse config
    backbone = hyperparams["model"]["backbone"]
    num_classes = hyperparams["model"]["num_classes"]
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
    
    ## get parameters
    root_dir = config.root_dir
    run_name = config.run_name

    ## set directories and paths
    data_dir = os.path.join(root_dir, "data", run_name)
    weights_dir = os.path.join(root_dir, "weights", run_name)
    log_dir = os.path.join(root_dir, "logs")
    run_dir = os.path.join(root_dir, "runs")
    swa_path = os.path.join(weights_dir, "swa")
    
    ## create directories
    os.makedirs(weights_dir,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(run_dir,exist_ok=True)

    ## scaler
    scaler = torch.cuda.amp.GradScaler()
    if not amp:
        scaler = None
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
    )
    model = utils.build_model(
        backbone,
        second_stage=(stage == "second"),
        num_classes=num_classes,
        ckpt_pretrained=swa_path,
    )
    
    ## configure multi-GPU system
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        
    ## activate cuda
    print(f"CUDA available: {torch.cuda.is_available()}")  
    model = model.cuda()

    optim = utils.build_optim(
        model, optimizer_params, scheduler_params, criterion_params
    )
    criterion, optimizer, scheduler, loss_optimizer = (
        optim["criterion"],
        optim["optimizer"],
        optim["scheduler"],
        optim["loss_optimizer"],
    )
    if ema:
        iters = len(loaders["train_features_loader"])
        ema_decay = ema_decay_per_epoch ** (1 / iters)
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    if loss_optimizer is not None and stage == 'second':
        raise ValueError('Loss optimizers should only be present for stage 1 training. Check your config file.') 
    
    # handle logging (regular logs, tensorboard, and weights)
    if run_name is None:
        run_name = "stage_{}_model_{}".format(
            stage, backbone
        )
        print(f"WARNING: No run-name set - using {run_name}!")
    writer = SummaryWriter(run_dir, filename_suffix= f"_{run_name}")
    logging_path = os.path.join(log_dir, f"{run_name}.log")
    logging.basicConfig(filename=logging_path, level=logging.INFO, filemode="w+")

    # epoch loop
    metric_best = 0
    for epoch in range(n_epochs):
        utils.add_to_logs(logging, "{}, epoch {}".format(time.ctime(), epoch))

        start_training_time = time.time()
        if stage == "first":
            train_metrics = utils.train_epoch_constructive(
                loaders["train_supcon_loader"], model, criterion, optimizer, scaler, ema, loss_optimizer
            )
        else:
            train_metrics = utils.train_epoch_ce(
                loaders["train_features_loader"],
                model,
                criterion,
                optimizer,
                scaler,
                ema,
            )
        end_training_time = time.time()

        if ema:
            copy_of_model_parameters = utils.copy_parameters_from_model(model)
            ema.copy_to(model.parameters())

        start_validation_time = time.time()
        if stage == "first":
            valid_metrics_projection_head = utils.validation_constructive(
                loaders["valid_loader"], loaders["train_features_loader"], model, scaler
            )
            
            ## check for GPU parallelization
            if model.__class__.__name__ == "DataParallel":
                model_copy = model.module
            else:
                model_copy = model
            
            model_copy.use_projection_head(False)
            valid_metrics_encoder = utils.validation_constructive(
                loaders["valid_loader"], loaders["train_features_loader"], model_copy, scaler
            )
            model_copy.use_projection_head(True)
            
            print(
                "epoch {}, train time {:.2f} valid time {:.2f} train loss {:.2f}\nvalid acc dict projection head {}\nvalid acc dict encoder {}".format(
                    epoch,
                    end_training_time - start_training_time,
                    time.time() - start_validation_time,
                    train_metrics["loss"],
                    valid_metrics_projection_head,
                    valid_metrics_encoder,
                )
            )
            valid_metrics = valid_metrics_projection_head
        else:
            valid_metrics = utils.validation_ce(
                model, criterion, loaders["valid_loader"], scaler
            )
            print(
                "epoch {}, train time {:.2f} valid time {:.2f} train loss {:.2f}\n valid acc dict {}\n".format(
                    epoch,
                    end_training_time - start_training_time,
                    time.time() - start_validation_time,
                    train_metrics["loss"],
                    valid_metrics,
                )
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
                # in case valid metric is a list
                pass

        if stage == "first":
            utils.add_to_logs(
                logging,
                "Epoch {}, train loss: {:.4f}\nvalid metrics projection head: {}\nvalid metric encoder: {}".format(
                    epoch,
                    train_metrics["loss"],
                    valid_metrics_projection_head,
                    valid_metrics_encoder,
                ),
            )
        else:
            utils.add_to_logs(
                logging,
                "Epoch {}, train loss: {:.4f} valid metrics: {}".format(
                    epoch, train_metrics["loss"], valid_metrics
                ),
            )
        # check if the best value of metric changed. If so -> save the model
        if (
            valid_metrics[target_metric] > metric_best*0.99
        ):  # > 0 if wanting to save all models 
            utils.add_to_logs(
                logging,
                "{} increased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    target_metric, metric_best, valid_metrics[target_metric]
                ),
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(weights_dir, f"epoch{epoch}"),
            )
            metric_best = valid_metrics[target_metric]

        # if ema is used, go back to regular weights without ema
        if ema:
            utils.copy_parameters_to_model(copy_of_model_parameters, model)

        scheduler.step()

    writer.close()


if __name__ == "__main__":
            
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    train(args.config_path)

