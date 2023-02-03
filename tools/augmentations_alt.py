import albumentations as A
import copy
from albumentations import pytorch as AT

def get_transforms(config, key='transforms', norm=True, valid=False):
    img_size = config.get('img_size', 224)
    aug = get_aug_from_config(config.get('augmentations', {}).get(key, None))
    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.NoOp() if valid else aug,
        A.Normalize() if norm else A.NoOp(),
        AT.ToTensorV2()
    ])

def get_aug_from_config(config):
    config = copy.deepcopy(config) or None

    if isinstance(config, str):
        return get_albumentation_class(config)()
    elif isinstance(config, list):
        return A.Compose([get_aug_from_config(c) for c in config])
    elif isinstance(config, dict):
        name = list(config.keys())[0]
        args = config[name].get("args", [])
        config = {**config[name], **config.get(name, {})}
        return get_albumentation_class(name)(*args, **config)
    return A.NoOp()

def get_albumentation_class(name: str):
    try:
        return getattr(A, name)
    except AttributeError:
        return getattr(A.pytorch, name)