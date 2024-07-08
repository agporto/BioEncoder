<div align="center">
    <p><img src="https://github.com/agporto/BioEncoder/raw/main/assets/bioencoder_logo.png" width="300"></p>
</div>

# BioEncoder

BioEncoder is a tool box for image classification and trait discovery in organismal biology. It relies on image classification models trained using metric learning to learn species trait data  (i.e., features) from images. This implementation is based on [SupCon](https://github.com/ivanpanshin/SupCon-Framework) and [timm-vis](https://github.com/novice03/timm-vis). 

Preprint on BioRxiv: [https://doi.org/10.1101/2024.04.03.587987]( https://doi.org/10.1101/2024.04.03.587987)

## Features

- Taxon-agnostic dataloaders (making it applicable to any dataset - not just biological ones)
- Support of [timm models](https://github.com/rwightman/pytorch-image-models), and [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)
- Access to state-of-the-art metric losses, such as [Supcon](https://arxiv.org/abs/2004.11362) and  [Sub-center ArcFace](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf).
- [Exponential Moving Average](https://github.com/fadel/pytorch_ema) for stable training, and Stochastic Moving Average for better generalization and performance.
- [LRFinder](https://github.com/davidtvs/pytorch-lr-finder) for the second stage of the training.
- Easy customization of hyperparameters, including augmentations, through `YAML` configs (check the [config-templates](config-templates) folder for examples)
- Custom augmentations techniques via [albumentations](https://github.com/albumentations-team/albumentations)
- TensorBoard logs and checkpoints (soon to come: WandB integration)
- Streamlit app with rich model visualizations (e.g., [Grad-CAM](https://arxiv.org/abs/1610.02391))
- Interactive [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) plots using [Bokeh](https://bokeh.org/)

<div align="center">
    <p><img src="https://github.com/agporto/BioEncoder/raw/main/assets/bioencoder-interactive-plot.gif" width="500"></p>
</div>

## Quickstart

(for more detailed information consult [the help files](help))

1\. Install BioEncoder (into a virtual environment with pytorch/CUDA): 
````
pip install bioencoder
````

2\. Download example dataset from the data repo: [https://zenodo.org/records/10909614/files/BioEncoder-data.zip](https://zenodo.org/records/10909614/files/BioEncoder-data.zip?download=1&preview=1). 
This archive contains the images and configuration files needed for step 3/4, as well as the final model checkpoints and a script to reproduce the results and figures presented in the paper. To play around with theinteractive figures and the model explorer you can also skip the training / SWA steps. 

3\. Start interactive session (e.g., in Spyder or VS code) and run the following commands one by one:

```python
## use "overwrite=True to redo a step

import bioencoder

## global setup
bioencoder.configure(root_dir=r"~/bioencoder_wd", run_name="v1")

## split dataset
bioencoder.split_dataset(image_dir=r"~/Downloads/damselflies-aligned-trai_val", max_ratio=6, random_seed=42, val_percent=0.1, min_per_class=20)

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
bioencoder.inference(config_path="bioencoder_configs/inference.yml", image="path/to/image.jpg")

```
4\. Alternatively, you can directly use the command line interface: 

```python
## use the flag "--overwrite" to redo a step

bioencoder_configure --root-dir "~/bioencoder_wd" --run-name v1
bioencoder_split_dataset --image-dir "~/Downloads/damselflies-aligned-trai_val" --max-ratio 6 --random-seed 42
bioencoder_train --config-path "bioencoder_configs/train_stage1.yml"
bioencoder_swa --config-path "bioencoder_configs/swa_stage1.yml"
bioencoder_interactive_plots --config-path "bioencoder_configs/plot_stage1.yml"
bioencoder_model_explorer --config-path "bioencoder_configs/explore_stage1.yml"
bioencoder_lr_finder --config-path "bioencoder_configs/lr_finder.yml"
bioencoder_train --config-path "bioencoder_configs/train_stage2.yml"
bioencoder_swa --config-path "bioencoder_configs/swa_stage2.yml"
bioencoder_model_explorer --config-path "bioencoder_configs/explore_stage2.yml"
bioencoder_inference --config-path "bioencoder_configs/inference.yml" --image "path/to/image.jpg"

```

## Citation

Please cite BioEncoder as follows:

```bibtex

@UNPUBLISHED{Luerig2024-ov,
  title    = "{BioEncoder}: a metric learning toolkit for comparative
              organismal biology",
  author   = "Luerig, Moritz D and Di Martino, Emanuela and Porto, Arthur",
  journal  = "bioRxiv",
  pages    = "2024.04.03.587987",
  month    =  apr,
  year     =  2024,
  language = "en",
  doi      = "10.1101/2024.04.03.587987"
}

```
