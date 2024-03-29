

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

## Get started 

1\. Download the example [image dataset](https://osf.io/download/gsd5z/) and the [yaml configuration](https://osf.io/download/wb5ga/) and unzip the files 

2\. Activate your environment

```
mamba activate bioencoder
```

3\. Run `bioencoder_configure` to set the bioencoder root dir and the run name - for example:
```
bioencoder_configure --root-dir bioencoder --run-name damselflies-example
```

You will get something like this:

```
(bioencoder) D:\temp\bioencoder-test>bioencoder_configure --root-dir bioencoder --run-name damselflies-example
BioEncoder config:
- root_dir: bioencoder
- root_dir_abs: D:\temp\bioencoder-test\bioencoder
- run_name: damselflies-example
Given your Python WD (D:\temp\bioencoder-test), the current BioEncoder run directory will be:
- D:\temp\bioencoder-test\bioencoder\damselflies-example
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
You will get something like this:
```
(bioencoder) D:\temp\bioencoder-test>bioencoder_split_dataset --image-dir data_raw\damselflies_aligned_resized
Number of images per class prior to balancing: [1099, 481, 137, 2402]
Minimum number of images per class: 137
Number of images per class after balancing: [959, 481, 137, 959]
0.1
Number of images per class reserved for validation: 63
Processing class androchrome...
Warning: androchrome contains more than 959 images. Only 959 images will be used.
Processing class infuscans...
Processing class infuscans-obsoleta...
Processing class male...
Warning: male contains more than 959 images. Only 959 images will be used.
```
You can have as many subdirectories as you need, depending on the number of classes in your classification task. The key is to make sure that all images belonging to the same class are stored in the same subdirectory. Also, you do not need to worry about image resolution at this stage. The images will be resized during training using the parameters defined within the `YAML` configuration files. If a single class contains an overwhelming percentage of images, please consider undersampling it.

2 ) Split into train and val sets 
 
To split the data into `train` and `val` sets, simply run :

```
python split_dataset.py --dataset /path/to/data_directory

```
The `split_dataset.py` script is a command line tool that takes as input a path to a root directory containing subdirectories of images, and splits the data into `train` and `val` sets. The `val` set contains 10% of the images, but they are evenly distributed across classes. This is done to ensure that validation metrics will not be influenced by the dominant classes. If a class does not contain enough images, that class is ignored (with a warning being displayed). The remaining 90% of images go to the `train` set.

This will create the following directory structure under the `project/` folder:

```
project/
    root_directory/
    bioencoder/
        train/
        val/
```
## Configuration

`Bioencoder` relies on `YAML` files to control the training process. Each `YAML` file contains several hyperparameters that can be modified according to users needs. These hyperparameters include:

- Model architecture
- Augmentations
- Loss functions
- etc..

Example config files can be found in the `configs/train` folder. These files provide a starting point for training `Bioencoder` models and can be modified to suit specific use cases.


## Training

To train the model, run the following commands:

```
python train.py --config_name configs/train/train_effnetb4_damselfly_stage1.yml
python swa.py --config_name configs/train/swa_effnetb4_damselfly_stage1.yml
python train.py --config_name configs/train/train_effnetb4_damselfly_stage2.yml
python swa.py --config_name configs/train/swa_effnetb4_damselfly_stage2.yml
```

In order to run `LRFinder` on the second stage of the training, run:
```
python learning_rate_finder.py --config_name configs/train/lr_finder_effnetb4_damselfly_stage2.yml
```

After that you can check the results of the training either in `logs` or `runs` directory. For example, in order to check tensorboard logs for the first stage of `Damselfly` training, run:
```
tensorboard --logdir runs/effnetb4_damselfly_stage1
```
## Visualizations 

This repo is supplied with [interactive](https://bokeh.org/) PCA and T-SNE visualizations so that you can check the embeddings you get after the training. To generate the interactive plot, use:
```
python interactive_plots.py --config_name configs/plot/plot_effnetb4_damselfly_stage1.yml
```
Similarly, we provide a model visualization playground, where individuals can get further insight into their data. To launch the app and explore the final classification model, simply use:
```
streamlit run model_explorer.py -- --ckpt_pretrained ./weights/effnetb4_damselfly_stage2/swa --stage second --num_classes 4
```
Model visualization techniques vary between `first` and `second` stage, so please make sure you select the appropriate ones.

## Custom datasets

`BioEncoder` was designed so that it could be easily applied to your custom dataset. Simply change the information on the configuration files (e.g., number of classes and dataset directory).
