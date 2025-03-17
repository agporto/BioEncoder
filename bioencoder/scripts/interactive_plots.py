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
        
    ## Load Bioencoder config
    root_dir, run_name = config.root_dir, config.run_name
    hyperparams = utils.load_yaml(config_path)
    
    ## Parse config
    backbone = hyperparams["model"]["backbone"]
    num_classes = hyperparams["model"].get("num_classes", None)
    checkpoint = hyperparams["model"].get("checkpoint", "swa")
    stage = hyperparams.get("model", {}).get("stage", "first")
    
    batch_sizes = {
        "train_batch_size": hyperparams.get("dataloaders", {}).get("train_batch_size"),
        "valid_batch_size": hyperparams.get("dataloaders", {}).get("valid_batch_size",1),
    }
    num_workers = hyperparams.get("dataloaders", {}).get("num_workers", 4)
    perplexity = hyperparams.get("perplexity", 30)

    plot_config = {
        "color_classes": hyperparams.get("color_classes", None),
        "color_map": hyperparams.get("color_map", "jet"),
        "plot_style": hyperparams.get("plot_style", 1),
        "point_size": hyperparams.get("point_size", 10),
    }
    
    
    ## Set up directories
    data_dir = os.path.join(root_dir, "data", run_name)
    plot_path = os.path.join(root_dir, "plots", run_name, f"embeddings_{run_name}.html")
    if not overwrite and not kwargs.get("ret_embeddings"):
        assert not os.path.isfile(plot_path), f"File exists: {plot_path}"
    
    ## Load model and set up
    print(f"Checkpoint: using {checkpoint} of {stage} stage")
    ckpt_pretrained = os.path.join(root_dir, "weights", run_name, stage, checkpoint)
    utils.set_seed()
    transforms = utils.build_transforms(hyperparams)
    loaders = utils.build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=(stage == "second"))
    model = utils.build_model(backbone, second_stage=(stage == "second"), num_classes=num_classes, ckpt_pretrained=ckpt_pretrained).cuda()
    model.use_projection_head(False)
    model.eval()
    
    ## Determine which embeddings to compute
    embeddings, labels, rel_paths = [], [], []
    
    ## val batch size cant be zero
    embeddings_val, labels_val = utils.compute_embeddings(loaders["valid_loader"], model)
    if len(embeddings_val) < len(loaders["valid_loader"].dataset.imgs):
        missed_imgs = len(loaders["valid_loader"].dataset.imgs) - len(embeddings_val)
        print(f"Warning: missed {missed_imgs} images because batch size was not a multiple of validation dataset size.")
    rel_paths_val = [item[0][len(root_dir) + 1:] for item in loaders["valid_loader"].dataset.imgs[:len(embeddings_val)]]
    embeddings.extend(embeddings_val)
    labels.extend(labels_val)
    rel_paths.extend(rel_paths_val)
    
    ## train set embeddings
    if batch_sizes["train_batch_size"] is not None:
        embeddings_train, labels_train = utils.compute_embeddings(loaders["train_loader"], model)
        if len(embeddings_train) < len(loaders["train_loader"].dataset.imgs):
            missed_imgs = len(loaders["train_loader"].dataset.imgs) - len(embeddings_train)
            print(f"Warning: missed {missed_imgs} images because batch size was not a multiple of training dataset size.")
        rel_paths_train = [item[0][len(root_dir) + 1:] for item in loaders["train_loader"].dataset.imgs[:len(embeddings_train)]]
        embeddings.extend(embeddings_train)
        labels.extend(labels_train)
        rel_paths.extend(rel_paths_train)
    
    ## Return embeddings without plotting
    if kwargs.get("ret_embeddings"):
        df = pd.DataFrame({"image_name": [os.path.basename(p) for p in rel_paths], "class": [os.path.basename(os.path.dirname(p)) for p in rel_paths]})
        return pd.concat([df, pd.DataFrame(embeddings)], axis=1)
        
    ## Reduce dimensionality
    if not perplexity:
        perplexity = min(100, len(embeddings) // 2)
        print(f"tSNE: using a perplexity value of {perplexity}")
    reduced_data, colnames, _ = helpers.embbedings_dimension_reductions(embeddings, perplexity)
    
    ## make plot
    df = pd.DataFrame(reduced_data, columns=colnames)
    df["paths"] = [os.path.join("..", "..", p) for p in rel_paths]
    df["class"], df["class_str"] = labels, [os.path.basename(os.path.dirname(p)) for p in rel_paths]
    df["dataset"] = df["paths"].apply(lambda x: "validation" if "/val/" in x else "train")
        
    helpers.bokeh_plot(df, out_path=plot_path, **plot_config)

    
def cli():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, help="Path to the YAML configuration file to create interactive plots.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files without asking.")
    args = parser.parse_args()

    interactive_plots_cli = utils.restore_config(interactive_plots)
    interactive_plots_cli(args.config_path, overwrite=args.overwrite)
    
    
if __name__ == "__main__":
    
    cli()
