import os
import argparse
import yaml
import pandas as pd
import torch

from pathlib import Path

from bioencoder.core import utils
from bioencoder.vis import helpers

def interactive_plots(    
        config_path, 
        **kwargs,
):
    
    ## load bioencoer config
    config = utils.load_config(kwargs.get("bioencoder_config_path"))
    root_dir = config.root_dir
    run_name = config.run_name
    
    ## load config
    hyperparams = utils.load_yaml(config_path)
    
    ## parse config
    backbone = hyperparams["model"]["backbone"]
    num_classes = hyperparams["model"]["num_classes"]
    stage = hyperparams["model"]["stage"]
    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        "valid_batch_size": hyperparams["dataloaders"]["valid_batch_size"],
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]
    color_classes = hyperparams.get("color_classes")

    ## set cuda device
    cuda_device = kwargs.get("cuda_device",0)
    torch.cuda.set_device(cuda_device)
    print(f"Using CUDA device {cuda_device}")

    ## init scaler
    scaler = torch.cuda.amp.GradScaler()
    
    ## set up dirs
    data_dir = os.path.join(root_dir,"data",  run_name)
    plot_dir = os.path.join(root_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    ## load weights
    ckpt_pretrained = os.path.join(config.root_dir, "weights", run_name, stage, "swa")

    ## set random seed
    utils.set_seed()

    transforms = utils.build_transforms(hyperparams)
    loaders = utils.build_loaders(
        data_dir, transforms, batch_sizes, num_workers, second_stage=(stage == "second")
    )
    model = utils.build_model(
        backbone,
        second_stage=(stage == "second"),
        num_classes=num_classes,
        ckpt_pretrained=ckpt_pretrained,
        cuda_device=cuda_device,
    ).cuda()
    model.use_projection_head(False)
    model.eval()

    embeddings_train, labels_train = utils.compute_embeddings(
        loaders["valid_loader"], model, scaler
    )
    
    if kwargs.get("ret_embeddings"):
        return embeddings_train, labels_train
    
    paths_train = [item[0] for item in loaders["valid_loader"].dataset.imgs]
    reduced_data, colnames, _ = helpers.embbedings_dimension_reductions(
        embeddings_train
    )

    df = pd.DataFrame(reduced_data, columns=colnames)
    df["paths"] = [
        os.path.join("..", "..", item) for item in paths_train
    ]
    df["class"] = labels_train
    df["class_str"] = [
        os.path.basename(os.path.dirname(item[0])) for item in loaders["valid_loader"].dataset.imgs
    ]
    
    if kwargs.get("ret_df"):
        return df
    
    helpers.bokeh_plot(df, out_path=os.path.join(plot_dir, f"{run_name}.html"), color_classes=color_classes)
    
    
def cli():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    
    interactive_plots(args.config_path)
    
    

if __name__ == "__main__":
    
    cli()
