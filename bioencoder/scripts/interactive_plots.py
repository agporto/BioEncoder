import os
import argparse
import numpy as np
import pandas as pd
import torch

from bioencoder.core import utils
from bioencoder.vis import helpers
from bioencoder import config


def _build_split_embeddings_df(rel_paths, embeddings, dataset_name):
    """
    Build metadata + embeddings DataFrame for a split with strict length alignment.
    """
    n_meta = len(rel_paths)
    n_embed = len(embeddings)
    n = min(n_meta, n_embed)
    if n == 0:
        raise ValueError(f"No samples available for split '{dataset_name}' (meta={n_meta}, embeddings={n_embed}).")
    if n_meta != n_embed:
        print(
            f"Warning: split '{dataset_name}' metadata/embedding length mismatch "
            f"(meta={n_meta}, embeddings={n_embed}); truncating to {n}.",
            flush=True,
        )

    rel_paths = rel_paths[:n]
    embeddings = embeddings[:n]
    df_meta = pd.DataFrame(
        {
            "image_name": [os.path.basename(p) for p in rel_paths],
            "class_str": [os.path.basename(os.path.dirname(p)) for p in rel_paths],
            "dataset": dataset_name,
        }
    )
    return pd.concat([df_meta, pd.DataFrame(embeddings)], axis=1)


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
    perplexity = hyperparams.get("perplexity")
    progress_bar = hyperparams.get("progress_bar", True)

    plot_config = {
        "color_classes": hyperparams.get("color_classes", None),
        "color_map": hyperparams.get("color_map", "jet"),
        "plot_style": hyperparams.get("plot_style", 1),
        "point_size": hyperparams.get("point_size", 10),
    }

    return_results = hyperparams.get("return_results", False)

    ## directories and file management
    data_dir = os.path.join(root_dir, "data", run_name)
    plot_dir = os.path.join(root_dir, "plots", run_name)
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "embeddings_interactive_plot.html")
    if not overwrite and not return_results:
        assert not os.path.isfile(plot_path), f"File already exists: {plot_path}"
    
    ## Load model and set up
    print(f"Checkpoint: using {checkpoint} of {stage} stage")
    ckpt_pretrained = os.path.join(root_dir, "weights", run_name, stage, checkpoint)
    seed = utils.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = utils.build_model(
        backbone,
        second_stage=(stage == "second"),
        num_classes=num_classes,
        ckpt_pretrained=ckpt_pretrained,
        cuda_device=device,
    ).to(device)
    model.use_projection_head(False)
    model.eval()
    
    ## prep computation
    transforms = utils.build_transforms(hyperparams)
    loaders = utils.build_loaders(
        data_dir, transforms, batch_sizes, num_workers, 
        second_stage=(stage == "second"), drop_last=False, shuffle_train=False)
    
    ## val set (always computed)
    embeddings_val, labels_val = utils.compute_embeddings(
        loaders["valid_loader"],
        model,
        device,
        progress_bar=progress_bar,
        progress_desc="Embeddings (val)",
    )
    rel_paths_val = [item[0][len(root_dir) + 1:] for item in loaders["valid_loader"].dataset.samples]
    df_embeddings = _build_split_embeddings_df(rel_paths_val, embeddings_val, "val")
    
    ## train set - skipped if zero batch size
    if batch_sizes["train_batch_size"] is not None:
        embeddings_train, labels_train = utils.compute_embeddings(
            loaders["train_loader"],
            model,
            device,
            progress_bar=progress_bar,
            progress_desc="Embeddings (train)",
        )
        rel_paths_train = [item[0][len(root_dir) + 1:] for item in loaders["train_loader"].dataset.samples]
        df_train = _build_split_embeddings_df(rel_paths_train, embeddings_train, "train")
        df_embeddings = pd.concat([df_embeddings, df_train], ignore_index=True)

    ## Stable order before reduction
    df_embeddings = df_embeddings.sort_values(by=["class_str", "dataset","image_name"]).reset_index(drop=True)

    ## Reduce dimensionality
    n_samples = len(df_embeddings)
    if n_samples < 3:
        raise ValueError(f"Need at least 3 samples for dimensionality reduction, got {n_samples}.")
    max_valid_perplexity = max(1.0, float(n_samples - 1))
    if perplexity is None:
        perplexity = min(30.0, max(5.0, (n_samples - 1) / 3))
    perplexity = min(float(perplexity), max_valid_perplexity - 1e-6)
    perplexity = max(perplexity, 1.0)
    print(f"tSNE: using perplexity {perplexity}")
    # Reduce on numeric embedding columns only
    embedding_matrix = df_embeddings.select_dtypes(include=[np.number])
    reduced_data, colnames, _ = helpers.embbedings_dimension_reductions(embedding_matrix, perplexity, seed)
    
    ## make plot
    df_plot = df_embeddings.select_dtypes(exclude=[np.number])
    df_plot['paths'] = df_plot.apply(lambda row: os.path.join(
        "..", "..", "data", run_name, row['dataset'], row['class_str'], row['image_name']), axis=1)
    df_plot["class"] = pd.Categorical(df_plot["class_str"]).codes
    df_plot = pd.concat([df_plot, pd.DataFrame(reduced_data, columns=colnames)], axis=1)

    helpers.bokeh_plot(df_plot, out_path=plot_path, **plot_config)

    # return embeddings and plot coords
    if return_results:
        return df_embeddings, df_plot



    
def cli():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Path to the YAML configuration file for interactive plots.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files without asking.")
    args = parser.parse_args()

    interactive_plots_cli = utils.restore_config(interactive_plots)
    interactive_plots_cli(args.config_path, overwrite=args.overwrite)
    
    
if __name__ == "__main__":
    
    cli()
