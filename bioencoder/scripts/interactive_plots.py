import os
import argparse
import numpy as np
import pandas as pd
import torch

from bioencoder.core import utils
from bioencoder.vis import helpers
from bioencoder import config

def interactive_plots(    
        config_path, 
        overwrite=False,
        **kwargs,
):

    """
    Generates interactive plots for visualizing high-dimensional embeddings of validation set.
    This function computes embeddings using a trained model, reduces their dimensionality,
    and plots them in an interactive plot saved as an HTML file. Optionally, it can also return
    embeddings data as a DataFrame (ret_embeddings=True).
    
    Parameters
    ----------
    config_path : str
        Path to the YAML file that contains settings for the model and training configurations.
        This configuration includes details on model architecture, data loaders, and other
        hyperparameters required for embedding computation.
    overwrite : bool, optional
        If True, allows the generated HTML plot file to overwrite existing files with the same name.
        If False, the function will check if the file exists and assert failure if it does. Default is False.
    
    
    Raises
    ------
    AssertionError
        If 'overwrite' is False and a plot file already exists at the specified location.
    FileNotFoundError
        If the configuration file does not exist.
    
    Examples
    --------
    To generate interactive plots for model embeddings:
        bioencoder.interactive_plots("/path/to/config.yaml")
    

    """
        
    ## load bioencoer config
    root_dir = config.root_dir
    run_name = config.run_name
    
    ## load config
    hyperparams = utils.load_yaml(config_path)
    
    ## parse config
    backbone = hyperparams["model"]["backbone"]
    num_classes = hyperparams["model"].get("num_classes", None)
    checkpoint = hyperparams["model"].get("checkpoint", "swa")
    stage = hyperparams["model"].get("stage", "first")
    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        "valid_batch_size": hyperparams["dataloaders"]["valid_batch_size"],
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]
    color_classes = hyperparams.get("color_classes", None)
    color_map = hyperparams.get("color_map", "jet")
    plot_style = hyperparams.get("plot_style", 1)
    point_size = hyperparams.get("point_size", 10)

    ## set up dirs
    data_dir = os.path.join(root_dir,"data",  run_name)
    plot_dir = os.path.join(root_dir, "plots", run_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    ## plot path
    plot_path = os.path.join(plot_dir, f"embeddings_{run_name}.html")
    if not overwrite and not kwargs.get("ret_embeddings"):
        assert not os.path.isfile(plot_path), f"File exists: {plot_path}"
    
    ## load weights
    print(f"Checkpoint: using {checkpoint} of {stage} stage")
    ckpt_pretrained = os.path.join(config.root_dir, "weights", run_name, stage, checkpoint)

    ## set random seed
    utils.set_seed()

    ## extract embeddings
    transforms = utils.build_transforms(hyperparams)
    loaders = utils.build_loaders(
        data_dir, transforms, batch_sizes, num_workers, second_stage=(stage == "second")
    )
    model = utils.build_model(
        backbone,
        second_stage=(stage == "second"),
        num_classes=num_classes,
        ckpt_pretrained=ckpt_pretrained,
    ).cuda()
    model.use_projection_head(False)
    model.eval()
    embeddings_train, labels_train = utils.compute_embeddings(
        loaders["valid_loader"], model
    )
    
    ## load dataset
    rel_paths_train = [item[0][len(root_dir) + 1:] for item in loaders["valid_loader"].dataset.imgs]
       
    ## return embeddings without plotting
    if kwargs.get("ret_embeddings"):
        
        df = pd.DataFrame([os.path.basename(item) for item in rel_paths_train], columns=["image_name"])
        df["class"] = [
            os.path.basename(os.path.dirname(item[0])) for item in loaders["valid_loader"].dataset.imgs
        ]
        return pd.concat([df, pd.DataFrame(embeddings_train)], axis=1)
    
    reduced_data, colnames, _ = helpers.embbedings_dimension_reductions(
        embeddings_train
    )       
    df = pd.DataFrame(reduced_data, columns=colnames)
    df["paths"] = [ os.path.join("..", "..", item) for item in rel_paths_train]
    df["class"] = labels_train
    df["class_str"] = [
        os.path.basename(os.path.dirname(item[0])) for item in loaders["valid_loader"].dataset.imgs
    ]
    
    ## check if color matches n classes
    if color_classes:
        assert len(np.unique(labels_train)) == len(color_classes), f"Number of classes is {len(np.unique(labels_train))}, but you only provided {len(color_classes)} colors"
    
    helpers.bokeh_plot(df, out_path=plot_path, color_map=color_map, color_classes=color_classes, 
                       plot_style=plot_style, point_size=point_size)
    
    
def cli():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, help="Path to the YAML configuration file to create interactive plots.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files without asking.")
    args = parser.parse_args()

    interactive_plots_cli = utils.restore_config(interactive_plots)
    interactive_plots_cli(args.config_path, overwrite=args.overwrite)
    
    
if __name__ == "__main__":
    
    cli()
