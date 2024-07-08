#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% imports

import argparse
import os
import matplotlib.pyplot as plt

from torch_lr_finder import LRFinder

from bioencoder import config, utils

#%% function

def lr_finder(    
        config_path, 
        overwrite=False,
        
        **kwargs,
):
    """
    Conducts a learning rate range test using the LRFinder to identify an optimal 
    learning rate for training a model. This function loads model configurations,
    initializes the model, performs the test, plots the results, and updates the 
    global configuration with the found learning rate.

    Parameters
    ----------
    config_path : str
        Path to the YAML file that contains the model and training configurations.
        This configuration specifies details such as model architecture, data loader 
        parameters, optimizer settings, and other hyperparameters necessary for training.
    overwrite : bool, optional
        If True, the generated plot will overwrite any existing file with the same name.
        If False, the function will assert the nonexistence of the file before saving the plot.
        Default is False.

    Raises
    ------
    AssertionError
        If 'overwrite' is False and a plot file with the intended name already exists.
    FileNotFoundError
        If the specified configuration file does not exist or is unreachable.

    Examples
    --------
    To run the learning rate finder with a specific configuration and enable overwriting
    of existing plots:
        bioencoder.lr_finder("/path/to/config.yaml", overwrite=True)

    Notes
    -----
    This function should be used with care, as the detected learning rates are not always at 
    the global, but only the local minimum for a range of losses. Best inspect the plot, 
    pick a suitable LR yourself, and supply it through the config file for stage 2.
    """
    
    ## load bioencoer config
    root_dir = config.root_dir
    run_name = config.run_name
    
    ## load config
    hyperparams = utils.load_yaml(config_path)
    
    ## parse config
    backbone = hyperparams["model"].get("backbone")
    num_classes = hyperparams["model"].get("num_classes")
    optimizer_params = hyperparams.get("optimizer")
    scheduler_params = None
    criterion_params = hyperparams.get("criterion")
    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"].get("train_batch_size", None),
        "valid_batch_size": hyperparams["dataloaders"].get("valid_batch_size", None),
    }
    num_workers = hyperparams["dataloaders"].get("num_workers", 8)
    num_iter = hyperparams.get("num_workers", 300)
    skip_start = hyperparams.get("skip_start", num_iter//10)
    skip_end = hyperparams.get("save_figure", num_iter//10)
    return_LR = hyperparams.get("return_LR", False)
    save_figure = hyperparams.get("save_figure", False)

    ## set up dirs
    data_dir = os.path.join(root_dir,"data",  run_name)
    plot_dir = os.path.join(root_dir, "plots", run_name)
    os.makedirs(plot_dir, exist_ok=True)

    ## construct file path
    plot_path = os.path.join(plot_dir, "LRfinder_{}_{}_{}.png".format(
        run_name,
        optimizer_params["name"],
        batch_sizes["train_batch_size"],
    ))
    if not overwrite:
        assert not os.path.isfile(plot_path), f"File exists: {plot_path}"
    
    ## load weights
    ckpt_pretrained = os.path.join(config.root_dir, "weights", run_name, "first", "swa")

    transforms = utils.build_transforms(hyperparams)
    loaders = utils.build_loaders(
        data_dir, transforms, batch_sizes, num_workers, second_stage=True
    )
    
    model = utils.build_model(
        backbone,
        second_stage=True,
        num_classes=num_classes,
        ckpt_pretrained=ckpt_pretrained,
    ).cuda()

    optim = utils.build_optim(
        model, optimizer_params, scheduler_params, criterion_params
    )
    criterion, optimizer, _ = (
        optim["criterion"],
        optim["optimizer"],
        optim["scheduler"],
    )
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(loaders["train_features_loader"], end_lr=1, num_iter=num_iter)

    fig, ax = plt.subplots()
    ax, lr = lr_finder.plot(ax=ax, skip_start=skip_start, skip_end=skip_end)
    config.lr = round(lr, 6)
    fig.suptitle(f"Suggested LR: {config.lr}" , fontsize=20)

    if save_figure:
        fig.savefig(plot_path)
        
    if return_LR:
        return config.lr

def cli():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",type=str, help="Path to the YAML configuration file that specifies hyperparameters for the LR finder.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files without asking.")
    args = parser.parse_args()

    lr_finder_cli = utils.restore_config(lr_finder)
    lr_finder_cli(args.config_path, overwrite=args.overwrite)

if __name__ == "__main__":
    
    cli()


