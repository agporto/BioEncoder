# Options

Below are the currently available options for model backbones, losses, optimizers, schedulers, and augmentations. Since models and augmentation definitions are provided through third party packages (timm and albumenations), we refer to their documnetation.

## Models

BioEncocder uses `timm` as a source for the latest SOTA model architectures. All models listed in the [timm GitHub repo](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv) are available for use as model backbones.

You can also list models directly via the [timm API](https://timm.fast.ai/models):

```python

import timm
timm.list_models()[:5]

```

## Losses

- `SupCon`: SupConLoss is a supervised contrastive loss function that encourages similar examples to be closer in the embedding space while pushing dissimilar examples apart.
- `LabelSmoothing`: LabelSmoothingLoss applies label smoothing to reduce overfitting by smoothing the labels, thereby preventing the model from becoming too confident in its predictions.
- `CrossEntropy`: nn.CrossEntropyLoss is a common loss function used in classification tasks, combining nn.LogSoftmax and nn.NLLLoss in one single class.
- `SubCenterArcFace`: SubCenterArcFaceLoss is a variant of ArcFace loss that introduces sub-centers for better intra-class compactness and inter-class discrepancy.
- `ArcFace`: ArcFaceLoss is used to enhance face recognition models by adding an angular margin to the softmax function, improving discriminative power.

## Optimizers

- `Adam`: The Adam optimizer is widely used due to its adaptive learning rate capabilities, combining the advantages of both RMSProp and AdaGrad.
- `AdamW`: AdamW is a variant of the Adam optimizer that decouples weight decay from the gradient updates, often leading to better performance with regularization.
- `SGD`: The Stochastic Gradient Descent optimizer is a simple yet effective optimization method, especially useful when combined with momentum and learning rate schedules.
- `LookAhead`: LookAhead is an advanced optimizer that improves the convergence of base optimizers by looking ahead to the average direction of a number of optimizer steps.
- `Ranger`: Ranger is a synergistic optimizer that combines the LookAhead technique with RAdam, aiming to enhance training stability and convergence speed.
- `RAdam`: The Rectified Adam optimizer rectifies the variance of the adaptive learning rate to provide a more stable and reliable training process.

## Schedulers

- `ReduceLROnPlateau`: Reduces the learning rate when a metric has stopped improving, which helps in adapting the learning rate dynamically based on the training progress.
- `CosineAnnealingLR`: Applies a cosine annealing schedule to the learning rate, decreasing it following the cosine function, which can lead to better convergence in some training scenarios.
- `CosineAnnealingWarmRestarts`: Similar to CosineAnnealingLR, but allows restarts of the learning rate at specified intervals, which can help the model escape from local minima.
- `CyclicLR`: Cyclically varies the learning rate between a lower and upper bound, which can help in avoiding local minima and potentially lead to faster convergence.
- `ExponentialLR`: Decreases the learning rate exponentially at each epoch, providing a simple yet effective way to decay the learning rate over time.


## Augmentations

BioEncoder supports `albumentations`-style augmentation for training - please refer to the [Albumentations-API](https://albumentations.ai/docs/api_reference/full_reference/?h=) for available [augmentations](https://albumentations.ai/docs/api_reference/full_reference/?h=#albumentations.augmentations) and [compositions](https://albumentations.ai/docs/api_reference/full_reference/?h=#albumentations.core.composition.Sequential). 

The example below combines `OneOf` and `Sequential` compositions. If only one is used, `args` can be omitted. `save_sample` saves a random sample of the supplied training images to <root_dir>/data/<run_name>/aug_sample for inspections. Together with the `dry_run=True` hyperparameter for the train script, this allows for experimenting with different augmentation pipelines.


    augmentations:
      sample_save: True
      sample_n: 10
      sample_seed: 42
      transforms:
        - OneOf:
            args:
              - RandomResizedCrop:
                  height: *size
                  width: *size
                  scale:  !!python/tuple [0.7,1]
              - Flip:
              - RandomRotate90:
            p: 1
        - Sequential:
            args:
              - MedianBlur:
                  blur_limit: 3
                  p: 0.3
              - ShiftScaleRotate:
                  p: 0.4
              - OpticalDistortion:
              - GridDistortion:
              - HueSaturationValue:
            p: 0.5

