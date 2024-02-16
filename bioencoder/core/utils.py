import random
import os
import numpy as np
import yaml

from copy import deepcopy
from dataclasses import make_dataclass, asdict

import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.metrics import f1_score, accuracy_score

from .losses import LOSSES
from .optimizers import OPTIMIZERS
from .schedulers import SCHEDULERS
from .models import SupConModel
from .datasets import create_dataset
from .augmentations import get_transforms


def load_yaml(yaml_path):
    
    with open(yaml_path, "r") as file:
        dictionary = yaml.full_load(file)

    return dictionary


def load_config(config_path=None):
    
    if not config_path:
        config_path = os.path.join(os.path.expanduser("~"), ".bioencoder")
    
    config = load_yaml(config_path)
    dataclass = make_dataclass(
        cls_name="config",
        fields=config
    )
    config_dataclass = dataclass(**config)
    
    return config_dataclass

def update_config(config, config_path=None):
    
    if not config_path:
        config_path = os.path.join(os.path.expanduser("~"), ".bioencoder")
    
    with open(config_path, 'w') as file:
        yaml.dump(config.__dict__, file, default_flow_style=False)


def set_seed(seed=42):
    """Set the random seed for the entire pipeline.

    Parameters:
    seed (int, optional): The seed value to set for all random number generators. Default is 42.

    """
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_to_logs(logging, message):
    """Add a message to a logging object.

    Parameters:
    logging (logging.Logger): The logging object to add the message to.
    message (str): The message to be logged.

    """
    logging.info(message)


def add_to_tensorboard_logs(writer, message, tag, index):
    """Add a scalar value to TensorBoard logs.

    Parameters:
    writer (tensorboardX.SummaryWriter): The TensorBoard writer to use for logging.
    message (float): The scalar value to log.
    tag (str): The tag for the scalar value.
    index (int): The global step at which to log the scalar value.

    """
    writer.add_scalar(tag, message, index)


class TwoCropTransform:
    """Create two crops of the same image.

    Attributes:
    crop_transform (callable): The crop transform to apply to the image to produce two crops.

    """
    def __init__(self, crop_transform):
        self.crop_transform = crop_transform

    def __call__(self, x):
        """Create two crops of the same image.

        Parameters:
        x (tensor): The input image.

        Returns:
        list of tensors: A list of two cropped images.

        """
        return [self.crop_transform(image=x), self.crop_transform(image=x)]


def build_transforms(config):
    """Build the train and validation transforms.

    Parameters:
    config (dict): The configuration containing the parameters for building the transforms.

    Returns:
    dict: A dictionary containing the train and validation transforms.

    """
    train_transforms = get_transforms(config)
    valid_transforms = get_transforms(config, valid=True)

    return {
        "train_transforms": train_transforms,
        "valid_transforms": valid_transforms
    }


def build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=False, is_supcon=False):
    """
    Build data loaders for training and validation.
    
    Parameters:
        data_dir (str): The directory where the dataset is located.
        transforms (dict): The transforms to be applied on the dataset.
        batch_sizes (dict): The batch sizes for training and validation.
        num_workers (int): The number of worker threads to use for loading data.
        second_stage (bool, optional): Whether to build loaders for second stage of training. 
                                       Defaults to False.
    
    Returns:
        dict: A dictionary containing the train and validation data loaders. If `second_stage` is False,
              it will also include the `train_supcon_loader`.
    """

    train_features_dataset = create_dataset(
        data_dir=data_dir, 
        train=True,
        transform=transforms['train_transforms'], 
        second_stage=True
    )

    valid_dataset = create_dataset(
        data_dir=data_dir, 
        train=False,
        transform=transforms['valid_transforms'], 
        second_stage=True
    )

    train_features_loader = torch.utils.data.DataLoader(
        train_features_dataset, 
        batch_size=batch_sizes['train_batch_size'], 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True
    )
        
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=batch_sizes['valid_batch_size'], 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    
    loaders = {
        'train_features_loader': train_features_loader, 
        'valid_loader': valid_loader
    }

    if not second_stage:
        train_supcon_dataset = create_dataset(
            data_dir=data_dir, 
            train=True,
            transform=TwoCropTransform(transforms['train_transforms']) if is_supcon else transforms['train_transforms'], 
            second_stage=False if is_supcon else True
        )

        train_supcon_loader = torch.utils.data.DataLoader(
            train_supcon_dataset, 
            batch_size=batch_sizes['train_batch_size'], 
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=True
        )

        loaders['train_supcon_loader'] = train_supcon_loader

    return loaders
    


def build_model(backbone, second_stage=False, num_classes=None, ckpt_pretrained=None, device=None):
    """
    Build and load the SupCon model

    Args:
    - backbone (str): The name of the backbone to use in the model.
    - second_stage (bool): Whether to build the model for the second stage of training.
    - num_classes (int, optional): The number of classes to predict. Defaults to None.
    - ckpt_pretrained (str, optional): The path to a checkpoint to load as pre-trained weights. Defaults to None.

    Returns:
    - model (torch.nn.Module): The SupCon model.
    """

    model = SupConModel(backbone=backbone, second_stage=second_stage, num_classes=num_classes)

    if ckpt_pretrained:
        model.load_state_dict(torch.load(ckpt_pretrained)['model_state_dict'], strict=False)

    return model


def build_optim(model, optimizer_params, scheduler_params, loss_params):
    """Build the optimizer, criterion, and scheduler for the model

    Args:
        model (torch.nn.Module): The model to optimize.
        optimizer_params (dict): The parameters for building the optimizer. The format is:
            {"name": str, "params": dict}. Defined in the config file.
        scheduler_params (dict, optional): The parameters for building the scheduler. The format is:
            {"name": str, "params": dict}. Defined in the config file.
        loss_params (dict): The parameters for building the loss function. The format is:
            {"name": str, "params": dict}. Defined in the config file.

    Returns:
        dict: The dictionary containing the built `criterion`, `optimizer`, and `scheduler`.
    """
    if 'params' in loss_params:
        criterion = LOSSES[loss_params['name']](**loss_params['params'])
    else:
        criterion = LOSSES[loss_params['name']]()
    
    if 'optimizer' in loss_params:
        loss_optimizer =  OPTIMIZERS[loss_params["optimizer"]["name"]](criterion.parameters(), **loss_params["optimizer"]["params"])
    else:
        loss_optimizer = None

    optimizer = OPTIMIZERS[optimizer_params["name"]](model.parameters(), **optimizer_params["params"])

    if scheduler_params:
        scheduler = SCHEDULERS[scheduler_params["name"]](optimizer, **scheduler_params["params"])
    else:
        scheduler = None

    return {"criterion": criterion, "optimizer": optimizer, "scheduler": scheduler, "loss_optimizer": loss_optimizer}


def compute_embeddings(loader, model, scaler):
    """Computes the embeddings and corresponding labels for a dataset.

    Parameters:
        loader (torch.utils.data.DataLoader): DataLoader that provides images and labels.
        model (torch.nn.Module): Neural network model used to compute the embeddings.
        scaler (torch.cuda.amp.autocast): Autocast context manager used to perform mixed-precision training.

    Returns:
        tuple: A tuple containing:
            np.ndarray: The embeddings computed by the model, of shape (num_samples, embedding_size).
            np.ndarray: The corresponding labels, of shape (num_samples,).
    """
    total_embeddings = None
    total_labels = None

    for images, labels in loader:
        images = images.cuda()
        if scaler:
            with torch.cuda.amp.autocast():
                embed = model(images)
        else:
            embed = model(images)
        if total_embeddings is None:
            total_embeddings = embed.detach().cpu()
            total_labels = labels.detach().cpu()
        else:
            total_embeddings = torch.cat((total_embeddings, embed.detach().cpu()))
            total_labels = torch.cat((total_labels, labels.detach().cpu()))

        del images, labels, embed

    torch.cuda.empty_cache()

    return np.float32(total_embeddings), np.uint8(total_labels)


def train_epoch_constructive(train_loader, model, criterion, optimizer, scaler, ema, loss_optimizer):
    """
    Trains the `model` on the data from the `train_loader` for one epoch. The loss function is defined by `criterion` and
    the optimization algorithm is defined by `optimizer`. The training process can also be scaled using the `scaler` and
    the `ema` (exponential moving average) can be applied to the model's parameters.

    Parameters:
    - train_loader (torch.utils.data.DataLoader): The data loader that provides the training data.
    - model (torch.nn.Module): The model that will be trained.
    - criterion (torch.nn.Module): The loss function to be used for training.
    - optimizer (torch.optim.Optimizer): The optimization algorithm to be used for training.
    - scaler (torch.cuda.amp.GradScaler, optional): The scaler used for gradient scaling in case of mixed precision training.
    - ema (ExponentialMovingAverage, optional): If provided, the exponential moving average to be applied to the model's parameters.

    Returns:
    - dict: A dictionary containing the mean loss over all training batches.
    """
    model.train()
    train_loss = []
    loss_optimization = False if loss_optimizer is None else True

    for idx, (images, labels) in enumerate(train_loader):
        if loss_optimization:
            images, labels = images.cuda(), labels.cuda()
        else:
            images = torch.cat([images[0]['image'], images[1]['image']], dim=0).cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

        if scaler:
            with torch.cuda.amp.autocast():
                embed = model(images)
                if not loss_optimization:
                    f1, f2 = torch.split(embed, [bsz, bsz], dim=0)
                    embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = criterion(embed, labels)

        else:
            embed = model(images)
            if not loss_optimization:
                f1, f2 = torch.split(embed, [bsz, bsz], dim=0)
                embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(embed, labels)


        del images, labels, embed
        torch.cuda.empty_cache()

        train_loss.append(loss.item())

        optimizer.zero_grad()
        if loss_optimization:
            loss_optimizer.zero_grad()

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if loss_optimization:
                scaler.step(loss_optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            if loss_optimization:
                loss_optimizer.step()

        if ema:
            ema.update(model.parameters())

    return {'loss': np.mean(train_loss)}


def validation_constructive(valid_loader, train_loader, model, scaler):
    """
    This function performs the validation step of the constructive learning algorithm. 

    Parameters:
        valid_loader (torch.utils.data.DataLoader): DataLoader containing the validation data.
        train_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
        model (torch.nn.Module): The model being trained.
        scaler (torch.cuda.amp.GradScaler): The scaler used for gradient scaling in case of mixed precision training.

    Returns:
        acc_dict (dict): A dictionary containing the accuracy metrics, computed using the `AccuracyCalculator` class.
    """

    calculator = AccuracyCalculator(k=1)
    model.eval()

    query_embeddings, query_labels = compute_embeddings(valid_loader, model, scaler)
    reference_embeddings, reference_labels = compute_embeddings(train_loader, model, scaler)

    acc_dict = calculator.get_accuracy(
        query_embeddings,
        query_labels,
        reference_embeddings,
        reference_labels,
    )

    del query_embeddings, query_labels, reference_embeddings, reference_labels
    torch.cuda.empty_cache()

    return acc_dict


def train_epoch_ce(train_loader, model, criterion, optimizer, scaler, ema):
    """
    Train the model for one epoch using cross-entropy loss.

    Parameters:
    train_loader (torch.utils.data.DataLoader): The data loader for the training data.
    model (torch.nn.Module): The model to be trained.
    criterion (torch.nn.Module): The loss function to be used for training.
    optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
    scaler (torch.cuda.amp.GradScaler): The scaler used for gradient scaling in case of mixed precision training.
    ema (Optional[torch.nn.Module]): The exponential moving average model.

    Returns:
    dict: A dictionary containing the mean loss over the epoch.
    """

    model.train()
    train_loss = []

    for batch_i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
                train_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        if ema:
            ema.update(model.parameters())

        del data, target, output
        torch.cuda.empty_cache()

    return {"loss": np.mean(train_loss)}


def validation_ce(model, criterion, valid_loader, scaler):
    """
    Validates the given model with cross entropy loss and calculates several evaluation metrics such as accuracy, F1 scores and F1 score macro.

    Parameters:
    model (torch.nn.Module): The model to be validated.
    criterion (torch.nn.modules.loss._Loss): The criterion to be used for validation, which is cross entropy loss in this case.
    valid_loader (torch.utils.data.DataLoader): The data loader for validation dataset.
    scaler (torch.cuda.amp.autocast.Autocast): Optional scaler for using automatic mixed precision (AMP).

    Returns:
    dict: A dictionary containing the validation loss, accuracy, F1 scores, and F1 score macro.

    """
    model.eval()
    val_loss = []
    y_pred, y_true = [], []

    for data, target in valid_loader:
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(data)
            else:
                output = model(data)

            if criterion:
                loss = criterion(output, target)
                val_loss.append(loss.item())

            pred = output.argmax(dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(target.cpu().numpy())

            del data, target, output
            torch.cuda.empty_cache()

    valid_loss = np.mean(val_loss)
    f1_scores = f1_score(y_true, y_pred, average=None)
    f1_score_macro = f1_score(y_true, y_pred, average='macro')
    acc_score = accuracy_score(y_true, y_pred)

    metrics = {"loss": valid_loss, "accuracy": acc_score, "f1_scores": f1_scores, 'f1_score_macro': f1_score_macro}
    return metrics


def copy_parameters_from_model(model):
    """
    Copy parameters from a PyTorch model.

    Args:
    model (nn.Module): The PyTorch model from which to copy parameters.

    Returns:
    list: A list of PyTorch tensors that represent the parameters of the model.
    """
    return [p.clone().detach() for p in model.parameters() if p.requires_grad]


def copy_parameters_to_model(params, model):
    """
    Copy the parameters from `params` to `model`.
    
    Parameters
    ----------
    params : List of torch.Tensor
        A list of parameters to be copied to the `model`.
    model : torch.nn.Module
        The target model where the parameters will be copied.
        
    Returns
    -------
    None
    """
    for s_param, param in zip(params, model.parameters()):
        if param.requires_grad:
            param.data.copy_(s_param.data)
