#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% imports

import argparse
import os
from collections import OrderedDict

import torch

from bioencoder import config, utils

#%% function

def swa(
    config_path, 
    **kwargs,
):
    """
    Applies Stochastic Weight Averaging (SWA) to improve model generalization by averaging multiple model checkpoints.

    Parameters
    ----------
    config_path : str
        Path to the YAML file containing training configurations including model, training parameters, 
        and data loader settings. This file dictates how the model and training processes should be structured.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file or model checkpoints are not found.
    ValueError
        If required configurations for stages or model details are missing or invalid.

    Examples
    --------
    To perform SWA on a model using a configuration file located at '/path/to/config.yaml':
        bioencoder.swa("/path/to/config.yaml")

    """
    
    ## load bioencoer config
    root_dir = config.root_dir
    run_name = config.run_name
        
    ## load config 
    hyperparams = utils.load_yaml(config_path)
    
    ## parse config
    backbone = hyperparams["model"]["backbone"]
    top_k_checkoints = hyperparams["model"]["top_k_checkpoints"]
    amp = hyperparams["train"]["amp"]
    stage = hyperparams["train"]["stage"]
    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        "valid_batch_size": hyperparams["dataloaders"]["valid_batch_size"],
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]

    ## manage second stage
    if stage == "second":
        num_classes = hyperparams["model"]["num_classes"]
    else:
        num_classes = None
        
    ## set directories
    data_dir = os.path.join(root_dir, "data", run_name)
    weights_dir = os.path.join(root_dir, "weights", run_name, stage)
    if os.path.exists(os.path.join(weights_dir, "swa")):
        os.remove(os.path.join(weights_dir, "swa"))

    ## scaler
    scaler = torch.cuda.amp.GradScaler()
    if not amp:
        scaler = None
    utils.set_seed()

    transforms = utils.build_transforms(hyperparams)
    loaders = utils.build_loaders(
        data_dir, transforms, batch_sizes, num_workers, second_stage=(stage == "second")
    )
    model = utils.build_model(
        backbone,
        second_stage=(stage == "second"),
        num_classes=num_classes,
        ckpt_pretrained=None,
    ).cuda()

    ## inspect epochs
    list_of_epochs = sorted([int(x.split("epoch")[1]) for x in os.listdir(weights_dir)])
    best_epochs = list_of_epochs[-top_k_checkoints::]

    checkpoints_paths = [
        "{}/{}{}".format(weights_dir, "epoch", epoch) for epoch in best_epochs
    ]    
    
    state_dicts = []
    for path in checkpoints_paths:
        state_dicts.append(torch.load(path)["model_state_dict"])

    average_dict = OrderedDict()
    for k in state_dicts[0].keys():
        average_dict[k] = sum([state_dict[k] for state_dict in state_dicts]) / len(
            state_dicts
        )

    torch.save({"model_state_dict": average_dict}, os.path.join(weights_dir, "swa"))
    model.load_state_dict(
        torch.load(os.path.join(weights_dir, "swa"))["model_state_dict"],
    )

    if stage == "first":
        valid_metrics = utils.validation_constructive(
            loaders["valid_loader"], loaders["train_features_loader"], model, scaler
        )
    else:
        valid_metrics = utils.validation_ce(
            model, None, loaders["valid_loader"], scaler
        )

    print("swa stage {} validation metrics: {}".format(stage, valid_metrics))
    
    
    
def cli():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",type=str, help="Path to the YAML configuration file that specifies hyperparameters for SWA.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files without asking.")
    args = parser.parse_args()


    swa_cli = utils.restore_config(swa)    
    swa_cli(args.config_path, overwrite=args.overwrite)
    

if __name__ == "__main__":
    
    cli()