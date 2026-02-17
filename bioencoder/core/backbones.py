import torchvision.models as models

if hasattr(models, "list_models"):
    _TORCHVISION_MODEL_NAMES = models.list_models()
else:
    _TORCHVISION_MODEL_NAMES = [
        name
        for name in dir(models)
        if not name.startswith("_") and callable(getattr(models, name))
    ]


def _get_model_builder(name):
    if hasattr(models, "get_model_builder"):
        return models.get_model_builder(name)
    return getattr(models, name)


# Build a dictionary of torchvision model builders only.
BACKBONES = {
    name: _get_model_builder(name)
    for name in _TORCHVISION_MODEL_NAMES
}
