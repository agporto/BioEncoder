

<p align="center"><img src="bioencoder_logo.png" width="300"></p>

# BioEncoder: A toolkit for imageomics

## About

`BioEncoder` is a rich toolset for image classification and trait discovery in organismal biology. It relies on image classification models trained using metric learning to learn species trait data  (i.e., features) from images. This implementation is based on [SupCon](https://github.com/ivanpanshin/SupCon-Framework) and [timm-vis](https://github.com/novice03/timm-vis). It includes the following features:

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


## Install

1\. Create a clean virtual environment 
```
mamba create -n bioencoder python=3.9
mamba activate bioencoder
```

2\. Install pytorch with CUDA. Go to https://pytorch.org/get-started/locally/ and choose your version - e.g.:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3\. Install bioencoder from pypi:
````
pip install bioencoder
````

## Get started (CLI mode)

(for detailed information consult [the help files](docs\01-detailed-readme.md))

1\. Download the example [image dataset](https://osf.io/download/gsd5z/) and the [yaml configuration](https://osf.io/download/wb5ga/) and unzip the files 

2\. Activate your environment

```
mamba activate bioencoder
```

3\. Run `bioencoder_configure` to set the bioencoder root dir and the run name - for example:
```
bioencoder_configure --root-dir bioencoder --run-name damselflies-example
```
This will create a root folder inside your project, where all relevant bioencoder data, logs, etc. will be stored - it will look like this

```
project-dir/
    bioencoder-root-dir/
        data
            <run-name>
                train
                    class_1/
                        image_1.jpg
                        image_2.jpg
                        ...
                    class_2/
                        image_1.jpg
                        image_2.jpg
                        ...
                    ...
                val
                    ...
        logs
            <run-name>
                <run-name>.log
        plots
            <run-name>.html
        runs
            <run-name>
                <run-name>_first
                    events.out.tfevents.1700919284.machine-name.15832.0
                <run-name>_second
                    events.out.tfevents.1700919284.machine-name.15832.1
        weights
            <run-name>
                first
                    epoch0
                    epoch1
                    ...
                    swa
                second
                    epoch0
                    epoch1
                    ...
                    swa
    ...
```                 

5\. Now run `bioencoder_split_dataset` to create the data folder containing training and validation images
```
bioencoder_split_dataset --image-dir data_raw\damselflies_aligned_resized
```

6\. Use `train_fullbody1_stage1.yml` to train the the first stage of the model:

```
bioencoder_train --config-path damselflies_config_files\train_fullbody1_stage1.yml"
```

7\. Continue as follows:

```
bioencoder_train --config-path damselflies_config_files\train_stage1.yml"
bioencoder_swa --config-path damselflies_config_files\swa_stage1.yml"
bioencoder_train --config-path damselflies_config_files\train_stage2.yml"
bioencoder_swa --config-path damselflies_config_files\swa_stage2.yml"
```
Inspect the training runs with 
```
tensorboard --logdir bioencoder\runs\damselflies
```

8\. Create interactive plots:

``` 
bioencoder_interactive_plots --config-path damselflies_config_files\plot_stage1.yml
```

9\. Run the model explorer

``` 
bioencoder_model_explorer --config-path config-path damselflies_config_files\explorer_stage1.yml
```

## Interactive mode

```
import os
import bioencoder

## set your project dir
os.chdir(r"D:\temp\bioencoder-test")

## set project dir and run name
bioencoder.configure(root_dir = r"bioencoder", run_name = "damselflies1")

## split dataset 
bioencoder.split_dataset(image_dir=r"data_raw\damselflies_aligned_resized")

## training / swa
bioencoder.train(config_path=r"damselflies_config_files\train_stage1.yml")
bioencoder.swa(config_path=r"damselflies_config_files\swa_stage1.yml")
bioencoder.train(config_path=r"damselflies_config_files\train_stage2.yml")
bioencoder.swa(config_path=r"damselflies_config_files\swa_stage2.yml")

## interactive plots
bioencoder.interactive_plots(config_path=r"damselflies_config_files\plot_stage1.yml")

## model explorer
bioencoder.model_explorer(config_path=r"damselflies_config_files\explore_stage1.yml")
```