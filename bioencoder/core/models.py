import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .backbones import BACKBONES


def _infer_features_dim(model: nn.Module) -> int:
    """Infer encoder output dimensionality from common classifier heads."""
    if hasattr(model, "num_features") and isinstance(model.num_features, int):
        return model.num_features

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc.in_features

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            return classifier.in_features
        if isinstance(classifier, nn.Sequential):
            for layer in reversed(list(classifier)):
                if isinstance(layer, nn.Linear):
                    return layer.in_features
                if isinstance(layer, nn.Conv2d):
                    return layer.in_channels

    if hasattr(model, "get_classifier"):
        try:
            classifier = model.get_classifier()
            if isinstance(classifier, nn.Linear):
                return classifier.in_features
        except Exception:
            pass

    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Linear):
            return layer.in_features
        if isinstance(layer, nn.Conv2d):
            return layer.out_channels

    raise TypeError("Failed to infer feature dimension for the selected backbone.")


def _build_torchvision_model(backbone: str):
    constructor = BACKBONES.get(backbone)
    if constructor is None:
        raise KeyError(backbone)
    try:
        return constructor(weights="DEFAULT")
    except TypeError:
        return constructor(pretrained=True)


def create_encoder(backbone:str):
    """
    Creates an encoder from the specified backbone.
    
    Args:
    - backbone: Name of the backbone to use. Must be one of the torchvision backbones, or a timm backbone.
                To use a timm backbone, add the prefix 'timm_' before the name.
                For instance, 'timm_resnet18' for ResNet18 from timm library.
                
    Returns:
    - Tuple of two elements:
        1) nn.Sequential: Encoder created from the specified backbone
        2) int: Features dimension of the encoder
        
    Raises:
    - RuntimeError: If the specified backbone is not found in the `BACKBONES` dictionary or the timm library.
    - TypeError: If the linear layer of the model can't be found.
    
    """
    try:
        if backbone.startswith("timm_"):
            model_name = backbone[len("timm_"):]
            model = timm.create_model(
                model_name=model_name,
                pretrained=True,
                num_classes=0,
                global_pool="avg",
            )
            features_dim = _infer_features_dim(model)
            return model, features_dim

        model = _build_torchvision_model(backbone)
        features_dim = _infer_features_dim(model)
        model = torch.nn.Sequential(*list(model.children())[:-1])
    except (RuntimeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Specify a correct backbone name. Use a valid torchvision model name, "
            "or prefix a valid timm model with 'timm_' (e.g., timm_resnet18)."
        ) from exc

    return model, features_dim


class BioEncoderModel(nn.Module):
    """
    This class implements the BioEncoder model based on supervised contrastive learning.
    The model consists of a feature encoder and either a classifier head for the second stage of 
    training or a projection head for the first stage of training.

    Parameters
    ----------
    backbone : str 
        Name of the backbone network to use for the feature encoder.
        projection_dim (int): Number of dimensions for the output of the projection head.
        second_stage (bool): Whether to use the classifier head for second stage of training.
        num_classes (int): Number of classes for the classification task, required if `second_stage` is True.


    """
    def __init__(self, backbone='resnet50', projection_dim=128, second_stage=False, num_classes=None):
        super(BioEncoderModel, self).__init__()
        self.encoder, self.features_dim = create_encoder(backbone)
        self.second_stage = second_stage
        self.projection_head = True
        self.projection_dim = projection_dim
        self.embed_dim = projection_dim

        if self.second_stage:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.classifier = nn.Linear(self.features_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.features_dim, self.features_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.features_dim, self.projection_dim))

    def use_projection_head(self, mode):
        """Toggle the use of the projection head.

        Parameters:
        mode (bool): Whether to use the projection head.

        """
        self.projection_head = mode
        if mode:
            self.embed_dim = self.projection_dim
        else:
            self.embed_dim = self.features_dim

    def forward(self, x):
        """Compute the forward pass through the network.

        Parameters:
        x (torch.Tensor): Input to the network, with shape (batch_size, 3, height, width).

        Returns:
        torch.Tensor: Output of the network, with shape (batch_size, num_classes) if `second_stage`
                      is True, or (batch_size, projection_dim) otherwise.

        """
        if self.second_stage:
            feat = self.encoder(x).view(x.size(0), -1)#.squeeze()
            return self.classifier(feat)
        else:
            feat = self.encoder(x).view(x.size(0), -1)#.squeeze()
            if self.projection_head:
                return F.normalize(self.head(feat), dim=1)
            else:
                return F.normalize(feat, dim=1)
