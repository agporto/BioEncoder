

# BioSupCon

In this implementations you will find:
- Augmentations with [albumentations](https://github.com/albumentations-team/albumentations)
- Hyperparameters (including augmentations) are moved to .yml configs
- [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) and [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) interactive plots using [bokeh](https://bokeh.org/)
- 2-step validation (for features before and after the projection head) using metrics like AMI, NMI, mAP, precision_at_1, etc with [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning).
- [Exponential Moving Average](https://github.com/fadel/pytorch_ema) for a more stable training, and Stochastic Moving Average for a better generalization and just overall performance.
- Automatic Mixed Precision (torch version) training in order to be able to train with a bigger batch size (roughly by a factor of 2).
- LabelSmoothing loss, and [LRFinder](https://github.com/davidtvs/pytorch-lr-finder) for the second stage of the training (FC).
- TensorBoard logs, checkpoints
- Support of [timm models](https://github.com/rwightman/pytorch-image-models), and [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)


## Install

1. Clone the repo:
```
git clone https://github.com/agporto/BioSupCon && cd BioSupCon/
```

2. Create a clean virtual environment 
```
python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
````
python -m pip install --upgrade pip
pip install -r requirements.txt
````

## Training

In order to execute `Damselfly` training run:
```
python train.py --config_name configs/train/train_effnetb4_damselfly_stage1.yml
python swa.py --config_name configs/train/swa_effnetb4_damselfly_stage1.yml
python train.py --config_name configs/train/train_effnetb4_damselfly_stage2.yml
python swa.py --config_name configs/train/swa_effnetb4_damselfly_stage2.yml
```

In order to run LRFinder on the second stage of the training, run:
```
python learning_rate_finder.py --config_name configs/train/lr_finder_effnetb4_damselfly_stage2.yml
```

After that you can check the results of the training either in `logs` or `runs` directory. For example, in order to check tensorboard logs for the first stage of `Damselfly` training, run:
```
tensorboard --logdir runs/effnetb4_damselfly_stage1
```
## Visualizations 

This repo is supplied with [bokeh](https://bokeh.org/) PCA and T-SNE visualizations so that you can check embeddings you get after the training. To generate the bokeh plot, use:
```
python interactive_plots.py --config_name configs/plots/plot_effnetb4_damselfly_stage1.yml
```

## Custom datasets

It's fairly easy to adapt this pipeline to custom datasets. Simply change the information on the configuration files (e.g., number of classes and dataset directory).
