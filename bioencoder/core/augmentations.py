import copy
import albumentations as A
from albumentations import pytorch as AT

def get_transforms(config, valid=False):
    """
    Return a transformation pipeline based on the provided configuration.

    Args:
        config (dict): Dictionary containing the configuration for the image transformations.
        valid (bool, optional): Indicates whether the transformation is for the validation set or not.
    
    Returns:
        albumentations.core.composition.Compose: The image transformation pipeline.
    """
    default_size = 224
    img_size = config.get('img_size', default_size)
    config_aug = config.get('augmentations', {})
    aug = get_aug_from_config(config_aug.get('transforms', []))

    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.NoOp() if valid else aug,
        A.Normalize(),
        AT.ToTensorV2()
    ])


def get_aug_from_config(config):
    """
    A helper function to create image augmentation pipeline based on a given configuration.
    
    Parameters:
        config (str, list, or dict): A string, list of strings, or dictionary representing the augmentation pipeline.
    
    Returns:
        aug (albumentations.augmentations.transforms): The constructed augmentation pipeline.
    """
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
    
    
# def get_aug_from_config2(config_aug):
#     """
#     A helper function to create image augmentation pipeline based on a given config_auguration.
    
#     Parameters:
#         config_aug (str, list, or dict): A string, list of strings, or dictionary representing the augmentation pipeline.
    
#     Returns:
#         aug (albumentations.augmentations.transforms): The constructed augmentation pipeline.
#     """
#     config_aug = copy.deepcopy(config_aug)

#     if config_aug is None:
#         return A.NoOp()
    
#     elif isinstance(config_aug, dict):
    
#         compose = config_aug.get("compose", {
#             "name": "Sequential",
#             "p1": 1})
          
#         transforms_inst = getattr(A, compose["name"])
#         transforms_inst(parse_transforms(config_aug.get("transforms", {})), p=compose["p1"])
        

# def parse_transforms(transforms):
#     transforms_list = []
#     for trans in transforms:
#         name = list(trans.keys())[0]
#         params = trans.get(name, {})
#         transforms_list.append(getattr(A, name)(**params))
        
#     return transforms_list