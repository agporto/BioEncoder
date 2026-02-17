import torch.optim as optim
import torch_optimizer as jettify_optim

LOOKAHEAD_CLASS = jettify_optim.Lookahead

OPTIMIZERS = {
    "Adam": optim.Adam,
    'AdamW': optim.AdamW,
    "SGD": optim.SGD,
    'Ranger': jettify_optim.Ranger,
    'RAdam': jettify_optim.RAdam,
}