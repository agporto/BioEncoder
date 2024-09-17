<div align="center">
    <p><img src="https://github.com/agporto/BioEncoder/raw/main/assets/bioencoder_logo.png" width="300"></p>
</div>

# BioEncoder

BioEncoder is a toolkit for supervised metric learning to i) learn and extract features from images, ii) enhance biological image classification, and iii) identify the features most relevant to classification. Designed for diverse and complex datasets, the package and the available metric losses can handle unbalanced classes and subtle phenotypic differences more effectively than non-metric approaches. The package includes taxon-agnostic data loaders, custom augmentation techniques, hyperparameter tuning through YAML configuration files, and rich model visualizations, providing a comprehensive solution for high-throughput analysis of biological images.

Read the paper: [https://onlinelibrary.wiley.com/doi/10.1111/ele.14495](https://onlinelibrary.wiley.com/doi/10.1111/ele.14495)

## Functionality

[>> Full list of available model architectures, losses, optimizers, schedulers, and augmentations <<](https://github.com/agporto/BioEncoder/blob/main/help/05-options.md)

- Taxon-agnostic dataloaders (making it applicable to any dataset - not just biological ones)
- Support of [timm models](https://github.com/rwightman/pytorch-image-models), and [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)
- Access to state-of-the-art metric losses, such as [Supcon](https://arxiv.org/abs/2004.11362) and [Sub-center ArcFace](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf).
- [Exponential Moving Average](https://github.com/fadel/pytorch_ema) for stable training, and Stochastic Moving Average for better generalization and performance.
- [LRFinder](https://github.com/davidtvs/pytorch-lr-finder) for the second stage of the training.
- Easy customization of hyperparameters, including augmentations, through `YAML` configs (check the [config-templates](config-templates) folder for examples)
- Custom augmentations techniques via [albumentations](https://github.com/albumentations-team/albumentations)
- TensorBoard logs and checkpoints (soon to come: WandB integration)
- Streamlit app with rich model visualizations (e.g., [Grad-CAM](https://arxiv.org/abs/1610.02391) and [timm-vis](https://github.com/novice03/timm-vis/blob/main/details.ipynb))
- Interactive [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) plots using [Bokeh](https://bokeh.org/)

<div align="center">
    <p><img src="https://github.com/agporto/BioEncoder/raw/main/assets/bioencoder-interactive-plot.gif" width="500"></p>
</div>

## Quickstart

[>> Comprehensive help files <<](help)

1\. Install BioEncoder (into a virtual environment with pytorch/CUDA): 
````
pip install bioencoder
````

2\. Get the [example dataset from the data repo](https://zenodo.org/records/13017212/files/BioEncoder-dataset.zip?download=1) and the config files by [downloading the git repo](https://github.com/agporto/BioEncoder/archive/refs/heads/main.zip), and extract both. 

3\. Start interactive session (e.g., in Spyder or VS code) and run the following commands one by one:

```python
## use "overwrite=True to redo a step

import bioencoder

## global setup (pick a target directory for all output that bioencoder generates, e.g. training dataset, model weights, etc.)
bioencoder.configure(root_dir=r"bioencoder_wd", run_name="v1")

## split dataset (the dataset you downloaded)
bioencoder.split_dataset(image_dir=r"damselflies-aligned-trai_val", max_ratio=6, random_seed=42, val_percent=0.1, min_per_class=20)

## train stage 1 
bioencoder.train(config_path=r"bioencoder_configs/train_stage1.yml")
bioencoder.swa(config_path=r"bioencoder_configs/swa_stage1.yml")

## explore embedding space and model from stage 1
bioencoder.interactive_plots(config_path=r"bioencoder_configs/plot_stage1.yml")
bioencoder.model_explorer(config_path=r"bioencoder_configs/explore_stage1.yml")

## (optional) learning rate finder for stage 2
bioencoder.lr_finder(config_path=r"bioencoder_configs/lr_finder.yml")

## train stage 2
bioencoder.train(config_path=r"bioencoder_configs/train_stage2.yml")
bioencoder.swa(config_path=r"bioencoder_configs/swa_stage2.yml")

## explore model from stage 2
bioencoder.model_explorer(config_path=r"bioencoder_configs/explore_stage2.yml")

## inference (stage 1 = embeddings, stage 2 = classification)
bioencoder.inference(config_path="bioencoder_configs/inference.yml", image="path/to/image.jpg" / np.array)

```
4\. Alternatively, you can directly use the command line interface: 

```python
## use the flag "--overwrite" to redo a step

bioencoder_configure --root-dir "~/bioencoder_wd" --run-name v1
bioencoder_split_dataset --image-dir "damselflies-aligned-trai_val" --max-ratio 6 --random-seed 42
bioencoder_train --config-path "bioencoder_configs/train_stage1.yml"
bioencoder_swa --config-path "bioencoder_configs/swa_stage1.yml"
bioencoder_interactive_plots --config-path "bioencoder_configs/plot_stage1.yml"
bioencoder_model_explorer --config-path "bioencoder_configs/explore_stage1.yml"
bioencoder_lr_finder --config-path "bioencoder_configs/lr_finder.yml"
bioencoder_train --config-path "bioencoder_configs/train_stage2.yml"
bioencoder_swa --config-path "bioencoder_configs/swa_stage2.yml"
bioencoder_model_explorer --config-path "bioencoder_configs/explore_stage2.yml"
bioencoder_inference --config-path "bioencoder_configs/inference.yml" --path "path/to/image.jpg"

```

## Citation

Please cite BioEncoder as follows:

```bibtex

@article{https://doi.org/10.1111/ele.14495,
author = {Lürig, Moritz D. and Di Martino, Emanuela and Porto, Arthur},
title = {BioEncoder: A metric learning toolkit for comparative organismal biology},
journal = {Ecology Letters},
volume = {27},
number = {8},
pages = {e14495},
keywords = {biodiversity, deep metric learning, feature space, machine learning, phenotypic differences, python package, species identification},
doi = {https://doi.org/10.1111/ele.14495},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/ele.14495},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/ele.14495},
year = {2024}
}



```
