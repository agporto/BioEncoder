import argparse
import os
from biosupcon import utils

import matplotlib.pyplot as plt
import yaml
from torch_lr_finder import LRFinder



def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="configs/train/lr_finder_supcon_effnet4_damselfly_stage2.yml",
    )

    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_name"], "r") as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams


if __name__ == "__main__":
    os.makedirs("lr_finder_plots", exist_ok=True)

    hyperparams = parse_config()
    backbone = hyperparams["model"]["backbone"]
    ckpt_pretrained = hyperparams["model"]["ckpt_pretrained"]
    num_classes = hyperparams['model']['num_classes']
    optimizer_params = hyperparams["optimizer"]
    scheduler_params = None
    criterion_params = hyperparams["criterion"]
    data_dir = hyperparams["dataset"]
    img_size = hyperparams["img_size"]
    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        "valid_batch_size": hyperparams["dataloaders"]["valid_batch_size"]
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]

    transforms = utils.build_transforms(hyperparams, second_stage=True)
    loaders = utils.build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=True)
    model = utils.build_model(backbone, second_stage=True, num_classes=num_classes,
                              ckpt_pretrained=ckpt_pretrained).cuda()

    optim = utils.build_optim(model, optimizer_params, scheduler_params, criterion_params)
    criterion, optimizer, scheduler = (
        optim["criterion"],
        optim["optimizer"],
        optim["scheduler"],
    )
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(loaders["train_features_loader"], end_lr=1, num_iter=300)
    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax)


    fig.savefig(
        "lr_finder_plots/supcon_{}_{}_bs_{}_stage_{}_lr_finder.png".format(
            optimizer_params["name"],
            data_dir.split("/")[-1],
            batch_sizes["train_batch_size"],
            'second'
        )
    )
