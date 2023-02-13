import torchvision.models as models

#Get list of torchvision models and build a dictionary of them
BACKBONES = {
    name: getattr(models, name)
    for name in models.list_models()
    if hasattr(models, name)
}
