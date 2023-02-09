import torch.optim as optim


SCHEDULERS = {
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "CyclicLR": optim.lr_scheduler.CyclicLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
}

