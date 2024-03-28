<p align="center"><img src="https://github.com/agporto/BioEncoder/raw/main/assets/bioencoder_logo.png" width="300"></p>

# BioEncoder: A toolkit for imageomics

## About

`BioEncoder` is a tool box for image classification and trait discovery in organismal biology. It relies on image classification models trained using metric learning to learn species trait data  (i.e., features) from images. This implementation is based on [SupCon](https://github.com/ivanpanshin/SupCon-Framework) and [timm-vis](https://github.com/novice03/timm-vis). It includes the following features (among others):

1) 





## Quickstart

(for more detailed information consult [the help files](help))

1\. Install BioEncoder (into a virtual environment with pytorch/CUDA): 
````
pip install bioencoder
````

2\. Download example dataset: https://osf.io/download/gsd5z/

3\. Start interactive session (e.g., in Spyder or VS code) and run:

```
import bioencoder

## global setup
bioencoder.configure(root_dir=r"bioencoder_wd", run_name="v1")

## split dataset
bioencoder.split_dataset(image_dir=r"~/Downloads/damselflies-aligned-trai_val", max_ratio=6, random_seed=42)

## train stage 1
bioencoder.train(config_path=r"bioencoder_configs/train_stage1.yml")
bioencoder.swa(config_path=r"bioencoder_configs/swa_stage1.yml")

## explore embedding space and model from stage 1
bioencoder.interactive_plots(config_path=r"bioencoder_configs/plot_stage1.yml")
bioencoder.model_explorer(config_path=r"bioencoder_configs/explore_stage1.yml")

## (optional) learning rate finder for stage 2
bioencoder.lr_finder(config_path=r"bioencoder_configs/lr_finder.yml")

## train stage 2
bioencoder.train(config_path=r"bioencoder_configs/train_stage2.yml", overwrite=True)
bioencoder.swa(config_path=r"bioencoder_configs/swa_stage2.yml")

## explore model from stage 2
bioencoder.model_explorer(config_path=r"bioencoder_configs/explore_stage2.yml")

```

## Features

- Taxon-agnostic dataloaders (making it applicable to any biological dataset)
- Streamlit app with rich model visualizations (e.g., [Grad-CAM](https://arxiv.org/abs/1610.02391))
- Custom augmentations techniques via [albumentations](https://github.com/albumentations-team/albumentations)
- Easy customization of hyperparameters, including augmentations, through `YAML` configs
- Interactive [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) plots using [Bokeh](https://bokeh.org/)
- [Exponential Moving Average](https://github.com/fadel/pytorch_ema) for stable training, and Stochastic Moving Average for better generalization and performance.
- Automatic data parallelization for multi-gpu training and automatic mixed precision for larger batch sizes (support varies across graphics cards)
- Access to state-of-the-art metric losses, such as [Supcon](https://arxiv.org/abs/2004.11362) and  [Sub-center ArcFace](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf).
- [LRFinder](https://github.com/davidtvs/pytorch-lr-finder) for the second stage of the training (FC).
- TensorBoard logs and checkpoints (soon, Weights-and-Biases integration)
- Support of [timm models](https://github.com/rwightman/pytorch-image-models), and [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)

## Citation

Please cite BioEncoder as follows:

```
@UNPUBLISHED{
    Lurig2024-pb,
    title     = "phenopype : A phenotyping pipeline for Python",
    author    = "L{\"u}rig, Moritz D and Di Martino, Emanuela and Porto, Arthur", 
    journal  = "bioRxiv",
    language  = "en",
    doi       = "xxxx"
}
```