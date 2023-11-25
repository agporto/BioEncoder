import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .backbones import BACKBONES


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
        if 'timm_' in backbone:
            backbone = backbone[5:]
            print(f"Using backbone: {backbone}")
            model = timm.create_model(model_name=backbone, pretrained=True)
        else:
            model = BACKBONES[backbone](pretrained=True)
    except RuntimeError or KeyError:
        raise RuntimeError('Specify the correct backbone name. Either one of torchvision backbones, or a timm backbone.'
                           'For timm - add prefix \'timm_\'. For instance, timm_resnet18')

    layers = torch.nn.Sequential(*list(model.children()))
    try:
        potential_last_layer = layers[-1]
        while not isinstance(potential_last_layer, nn.Linear):
            potential_last_layer = potential_last_layer[-1]
    except TypeError:
        raise TypeError('Failed to find the linear layer of the model')

    features_dim = potential_last_layer.in_features
    model = torch.nn.Sequential(*list(model.children())[:-1])

    return model, features_dim


class SupConModel(nn.Module):
    """
    This class implements the supervised contrastive learning model.
    The model consists of a feature encoder and either a classifier head for the second stage of 
    training or a projection head for the first stage of training.

    Parameters:
    backbone (str): Name of the backbone network to use for the feature encoder.
    projection_dim (int): Number of dimensions for the output of the projection head.
    second_stage (bool): Whether to use the classifier head for second stage of training.
    num_classes (int): Number of classes for the classification task, required if `second_stage` is True.


    """
    def __init__(self, backbone='resnet50', projection_dim=128, second_stage=False, num_classes=None):
        super(SupConModel, self).__init__()
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
