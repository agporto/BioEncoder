#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import os

import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from torch_lr_finder import LRFinder

from bioencoder.core import utils

#%%

def lr_finder(    
        config_path, 
        overwrite=False,
        **kwargs,
):
    """
    

    Parameters
    ----------
    config_path : TYPE
        DESCRIPTION.
    overwrite : TYPE, optional
        DESCRIPTION. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ## load bioencoer config
    config = utils.load_config(kwargs.get("bioencoder_config_path"))
    root_dir = config.root_dir
    run_name = config.run_name
    
    ## load config
    hyperparams = utils.load_yaml(config_path)
    
    ## parse config
    backbone = hyperparams["model"]["backbone"]
    num_classes = hyperparams["model"]["num_classes"]
    optimizer_params = hyperparams["optimizer"]
    scheduler_params = None
    criterion_params = hyperparams["criterion"]
    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        "valid_batch_size": hyperparams["dataloaders"]["valid_batch_size"],
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]
    
    ## set up dirs
    data_dir = os.path.join(root_dir,"data",  run_name)
    plot_dir = os.path.join(root_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    ## construct file path
    plot_path = os.path.join(plot_dir, "{}_lr_finder_supcon_{}_bs_{}.png".format(
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
    criterion, optimizer, scheduler = (
        optim["criterion"],
        optim["optimizer"],
        optim["scheduler"],
    )
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(loaders["train_features_loader"], end_lr=1, num_iter=300)
    fig, ax = plt.subplots()
    
    f = io.StringIO()
    with redirect_stdout(f):
        lr_finder.plot(ax=ax)
    s = f.getvalue()
    print_msg = s.split("\n")[1]
    print(print_msg)
    
    config.second_lr = print_msg.split(": ")[1]
    utils.update_config(config)
    
    fig.suptitle(print_msg, fontsize=20)

    fig.savefig(plot_path)

def cli():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
    )
    args = parser.parse_args()
    
    lr_finder(args.config_path)


if __name__ == "__main__":
    
    cli()


