# Options

## Models

- timm -> link (or options fun)

## Losses

SupCon: SupConLoss is a supervised contrastive loss function that encourages similar examples to be closer in the embedding space while pushing dissimilar examples apart.

LabelSmoothing: LabelSmoothingLoss applies label smoothing to reduce overfitting by smoothing the labels, thereby preventing the model from becoming too confident in its predictions.

CrossEntropy: nn.CrossEntropyLoss is a common loss function used in classification tasks, combining nn.LogSoftmax and nn.NLLLoss in one single class.

KLDiv: nn.KLDivLoss measures how one probability distribution diverges from a second, expected probability distribution.

SubCenterArcFace: SubCenterArcFaceLoss is a variant of ArcFace loss that introduces sub-centers for better intra-class compactness and inter-class discrepancy.

ArcFace: ArcFaceLoss is used to enhance face recognition models by adding an angular margin to the softmax function, improving discriminative power.

## Optimizers

Adam: The Adam optimizer is widely used due to its adaptive learning rate capabilities, combining the advantages of both RMSProp and AdaGrad.

AdamW: AdamW is a variant of the Adam optimizer that decouples weight decay from the gradient updates, often leading to better performance with regularization.

SGD: The Stochastic Gradient Descent optimizer is a simple yet effective optimization method, especially useful when combined with momentum and learning rate schedules.

LookAhead: LookAhead is an advanced optimizer that improves the convergence of base optimizers by looking ahead to the average direction of a number of optimizer steps.

Ranger: Ranger is a synergistic optimizer that combines the LookAhead technique with RAdam, aiming to enhance training stability and convergence speed.

RAdam: The Rectified Adam optimizer rectifies the variance of the adaptive learning rate to provide a more stable and reliable training process.

## Schedulers

ReduceLROnPlateau: Reduces the learning rate when a metric has stopped improving, which helps in adapting the learning rate dynamically based on the training progress.

CosineAnnealingLR: Applies a cosine annealing schedule to the learning rate, decreasing it following the cosine function, which can lead to better convergence in some training scenarios.

CosineAnnealingWarmRestarts: Similar to CosineAnnealingLR, but allows restarts of the learning rate at specified intervals, which can help the model escape from local minima.

CyclicLR: Cyclically varies the learning rate between a lower and upper bound, which can help in avoiding local minima and potentially lead to faster convergence.

ExponentialLR: Decreases the learning rate exponentially at each epoch, providing a simple yet effective way to decay the learning rate over time.

## Augmentations

- albumentations -> link (or options fun)
