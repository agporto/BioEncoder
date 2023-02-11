import os
from matplotlib import pyplot as plt
import torch
import torch.functional as F
import biosupcon
import albumentations as A
import albumentations.pytorch as AT
import pandas as pd
import argparse
import yaml

scaler = torch.cuda.amp.GradScaler()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="configs/plots/plot_effnetb4_damselfly_stage1.yml",
    )

    parser_args = parser.parse_args()
    basename = os.path.basename(parser_args.config_name)
    with open(vars(parser_args)["config_name"], "r") as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams, basename


if __name__ == "__main__":
    hyperparams, basename = parse_config()

    backbone = hyperparams["model"]["backbone"]
    ckpt_pretrained = hyperparams["model"]["ckpt_pretrained"]
    num_classes = hyperparams["model"]["num_classes"]
    stage = hyperparams["model"]["stage"]
    data_dir = hyperparams["dataset"]
    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        "valid_batch_size": hyperparams["dataloaders"]["valid_batch_size"],
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]
    img_size = hyperparams["img_size"]

    biosupcon.utils.set_seed()

    transforms = biosupcon.utils.build_transforms(hyperparams)
    loaders = biosupcon.utils.build_loaders(
        data_dir, transforms, batch_sizes, num_workers, second_stage=(stage == "second")
    )
    model = biosupcon.utils.build_model(
        backbone,
        second_stage=(stage == "second"),
        num_classes=num_classes,
        ckpt_pretrained=ckpt_pretrained,
    ).cuda()
    model.use_projection_head(False)
    model.eval()

    embeddings_train, labels_train = biosupcon.utils.compute_embeddings(
        loaders["valid_loader"], model, scaler
    )
    paths_train = [item[0] for item in loaders["valid_loader"].dataset.imgs]
    reduced_data, colnames, _ = biosupcon.vis.embbedings_dimension_reductions(
        embeddings_train
    )

    df = pd.DataFrame(reduced_data, columns=colnames)
    df["paths"] = paths_train
    df["class"] = labels_train
    df["class_str"] = [
        item[0].split("/")[-2] for item in loaders["valid_loader"].dataset.imgs
    ]
    os.makedirs("./plots", exist_ok=True)
    biosupcon.vis.bokeh_plot(df, out_path=f"./plots/{basename}.html")
