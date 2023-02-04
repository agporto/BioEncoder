import copy

import albumentations as A
from albumentations import pytorch as AT

def get_transforms(config, key='transforms', norm=True, valid=False):
    img_size = config.get('img_size', 224)
    aug = get_aug_from_config(config['augmentations'][key]) if key in config['augmentations'] else A.NoOp()

    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.NoOp() if valid else aug,
        A.Normalize() if norm else A.NoOp(),
        AT.ToTensorV2()
    ])


def get_aug_from_config(config):
    config = copy.deepcopy(config)

    if config is None:
        return A.NoOp()

    if isinstance(config, str):
        return getattr(A, config)()

    if isinstance(config, list):
        return A.Sequential([get_aug_from_config(c) for c in config], p = 1.0)

    name = list(config.keys())[0]
    config = config[name] if config[name] else {}

    args = config.pop("args", None)
    args = args if args is not None else []

    if name == "Sequential":
        return A.Sequential([get_aug_from_config(c) for c in args], **config)
    elif name == "OneOf":
        return A.OneOf([get_aug_from_config(c) for c in args], **config)
    elif name == "OneOrOther":
        return A.OneOrOther([get_aug_from_config(c) for c in args], **config)
    elif name == "SomeOf":
        return A.SomeOf([get_aug_from_config(c) for c in args], **config)
    else:
        return getattr(A, name)(*args, **config)