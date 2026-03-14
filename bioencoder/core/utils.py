import cv2
import importlib
import random
import os
import shutil
import numpy as np
import yaml
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from functools import wraps

import torch
import torch.distributed as dist
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.metrics import f1_score #, accuracy_score

from .losses import LOSSES
from .optimizers import OPTIMIZERS, LOOKAHEAD_CLASS
from .schedulers import SCHEDULERS
from .models import BioEncoderModel
from .datasets import create_dataset
from .augmentations import get_transforms
from bioencoder.vis import helpers


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0

def init_distributed(backend="nccl", local_rank=None):
    if local_rank is None:
        local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")



def teardown_distributed():
    if is_distributed():
        dist.destroy_process_group()

def save_yaml(dic, yaml_path):
    with open(yaml_path, 'w') as file:
        yaml.dump(dic, file, default_flow_style=False)

def load_yaml(yaml_path):
    
    with open(yaml_path, "r") as file:
        dictionary = yaml.full_load(file)

    return dictionary  

def restore_config(func):
    """
    Decorator to restore configuration from a YAML file and inject it into the bioencoder.config module.
    Needed only when using BioEncoder in CLI mode, NOT in interactive mode where config is loaded directly
    When the decorated function is called, the decorator will:
        
    1. Load the configuration from a predefined YAML file path.
    2. Import the bioencoder.config module.
    3. Update the attributes of the bioencoder.config module with the loaded configuration values.
    4. Execute the original function with the injected configuration.
   
    Notes
    -----
    - The decorator expects the configuration file to be located at '~/.bioencoder.yaml'.
    - The configuration file should be in YAML format.
    - The attributes in the YAML file must match the expected attributes in the bioencoder.config module.
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        config_path = os.path.expanduser("~/.bioencoder.yaml")  # Updated to load from YAML
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Global BioEncoder config not found at '{config_path}'. "
                "Run the configure CLI first (e.g., bioencoder_configure --root-dir <path> --run-name <name>)."
            )
        config = load_yaml(config_path)

        # Import the bioencoder config module and update its attributes
        config_module = importlib.import_module('bioencoder.config')
        for key, value in config.items():
            setattr(config_module, key, value)
                
        return func(*args, **kwargs)
    return wrapper


def load_model(
        ckpt_pretrained, 
        backbone, 
        num_classes, 
        stage,
        cuda_device
        ):
    device = cuda_device if isinstance(cuda_device, torch.device) else torch.device(cuda_device)
    model = build_model(
        backbone, second_stage=(stage == 'second'), 
        num_classes=num_classes, ckpt_pretrained=ckpt_pretrained, 
        cuda_device=device).to(device)
    model.use_projection_head((stage=='second'))
    model.eval()
    
    return model

def update_config(config, config_path=None):
    
    if not config_path:
        config_path = os.path.join(os.path.expanduser("~"), ".bioencoder.yaml")
    
    with open(config_path, 'w') as file:
        yaml.dump(config.__dict__, file, default_flow_style=False)


def set_seed(seed=42, rank_offset=0):
    """Set the random seed for the entire pipeline.

    Parameters:
    seed (int, optional): The seed value to set for all random number generators. Default is 42.

    """
    seed_value = int(seed) + int(rank_offset)
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    
    return seed_value

def pprint_fill_hbar(message, symbol="-", ret=True):
    try:
        # Try to get the terminal width
        terminal_width = os.get_terminal_size()[0] - len("%Y-%m-%d %H:%M:%S")
    except OSError:
        # Fallback width for headless environments
        terminal_width = 80  # Default width if terminal size can't be determined

    message_length = len(message)

    if message_length >= terminal_width:
        formatted_message = message
    else:
        bar_length = (terminal_width - message_length - 2) // 2
        horizontal_bar = symbol * bar_length
        formatted_message = f"{horizontal_bar} {message} {horizontal_bar}"
        residual = terminal_width - len(formatted_message)
        formatted_message = formatted_message + symbol * residual
        
    if not ret:
        print(formatted_message)
    else:
        return formatted_message
    
def zip_directory(directory, rel_to, zip):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path_abs = os.path.join(root, file)
            file_path_rel = os.path.relpath(file_path_abs, rel_to)
            zip.write(file_path_abs, file_path_rel)



def add_to_tensorboard_logs(writer, message, tag, index):
    """Add a scalar value to TensorBoard logs.

    Parameters:
    writer (tensorboardX.SummaryWriter): The TensorBoard writer to use for logging.
    message (float): The scalar value to log.
    tag (str): The tag for the scalar value.
    index (int): The global step at which to log the scalar value.

    """
    if writer is not None:
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
    valid_transforms = get_transforms(config, no_aug=True)

    return {
        "train_transforms": train_transforms,
        "valid_transforms": valid_transforms
    }


def build_loaders(data_dir, transforms, batch_sizes, num_workers, 
                  second_stage=False, is_supcon=False,
                  shuffle_train=True, drop_last=True,
                  train_sampler=None, valid_sampler=None, train_supcon_sampler=None,
                  distributed=False, rank=0, world_size=1):
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

    if distributed:
        train_sampler = DistributedSampler(
            train_features_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle_train,
            drop_last=drop_last,
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = torch.utils.data.DataLoader(
        train_features_dataset, 
        batch_size=batch_sizes['train_batch_size'], 
        shuffle=shuffle_train if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last and batch_sizes['train_batch_size'] is not None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=batch_sizes['valid_batch_size'], 
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(),
        # Keep all validation samples for unbiased validation metrics.
        drop_last=False,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    
    loaders = {
        'train_loader': train_loader, 
        'valid_loader': valid_loader
    }

    if not second_stage:
        train_supcon_dataset = create_dataset(
            data_dir=data_dir, 
            train=True,
            transform=TwoCropTransform(transforms['train_transforms']) if is_supcon else transforms['train_transforms'], 
            second_stage=False if is_supcon else True
        )

        if distributed:
            train_supcon_sampler = DistributedSampler(
                train_supcon_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=drop_last,
            )

        train_supcon_loader = torch.utils.data.DataLoader(
            train_supcon_dataset, 
            batch_size=batch_sizes['train_batch_size'], 
            shuffle=True if train_supcon_sampler is None else False,
            sampler=train_supcon_sampler,
            num_workers=num_workers, 
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last and batch_sizes['train_batch_size'] is not None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            persistent_workers=(num_workers > 0),
        )

        loaders['train_supcon_loader'] = train_supcon_loader

    return loaders
    


def build_model(backbone, second_stage=False, num_classes=None, ckpt_pretrained=None, cuda_device=0):
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

    model = BioEncoderModel(backbone=backbone, second_stage=second_stage, num_classes=num_classes)

    if ckpt_pretrained:
        map_location = cuda_device if isinstance(cuda_device, torch.device) else torch.device(cuda_device)
        model.load_state_dict(torch.load(ckpt_pretrained, map_location=map_location)['model_state_dict'], strict=False)

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
    
    def create_optimizer(parameters, spec):
        name = spec["name"]
        params = spec.get("params", {})
        if name == "LookAhead":
            cfg = params.copy()
            base_name = cfg.pop("base_optimizer", "Adam")
            explicit_base_params = cfg.pop("base_params", None)
            wrapper_keys = {"k", "alpha", "pullback_momentum"}
            wrapper_params = {k: cfg.pop(k) for k in list(cfg.keys()) if k in wrapper_keys}
            if explicit_base_params is None:
                base_params = cfg
            else:
                base_params = {**cfg, **explicit_base_params}

            if base_name not in OPTIMIZERS:
                raise ValueError(
                    f"LookAhead base_optimizer '{base_name}' is not supported. "
                    f"Choose one of: {list(OPTIMIZERS.keys())}"
                )
            base_optimizer = OPTIMIZERS[base_name](parameters, **base_params)
            return LOOKAHEAD_CLASS(base_optimizer, **wrapper_params)

        if name not in OPTIMIZERS:
            raise ValueError(f"Optimizer '{name}' is not supported. Choose one of: {list(OPTIMIZERS.keys()) + ['LookAhead']}")
        return OPTIMIZERS[name](parameters, **params)

    if 'optimizer' in loss_params:
        loss_optimizer = create_optimizer(criterion.parameters(), loss_params["optimizer"])
    else:
        loss_optimizer = None

    optimizer = create_optimizer(model.parameters(), optimizer_params)

    if scheduler_params:
        scheduler = SCHEDULERS[scheduler_params["name"]](optimizer, **scheduler_params["params"])
    else:
        scheduler = None

    return {"criterion": criterion, "optimizer": optimizer, "scheduler": scheduler, "loss_optimizer": loss_optimizer}


def _all_gather_cat(tensor):
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = int(torch.stack(size_list).max().item())

    if tensor.shape[0] < max_size:
        pad_shape = (max_size - tensor.shape[0],) + tensor.shape[1:]
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad], dim=0)

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    outputs = []
    for idx, chunk in enumerate(gathered):
        outputs.append(chunk[: int(size_list[idx].item())])
    return torch.cat(outputs, dim=0)


def compute_embeddings(loader, model, device, scaler=None, progress_bar=False, progress_desc=None):
    """Computes the embeddings and corresponding labels for a dataset.

    Parameters:
        loader (torch.utils.data.DataLoader): DataLoader that provides images and labels.
        model (torch.nn.Module): Neural network model used to compute the embeddings.
        scaler (torch.amp.autocast): Autocast context manager used to perform mixed-precision training.

    Returns:
        tuple: A tuple containing:
            np.ndarray: The embeddings computed by the model, of shape (num_samples, embedding_size).
            np.ndarray: The corresponding labels, of shape (num_samples,).
    """
    total_embeddings = None
    total_labels = None

    pbar = None
    if progress_bar and is_main_process():
        pbar = tqdm(total=len(loader), desc=progress_desc or "Validation", dynamic_ncols=True, leave=False)

    try:
        for images, labels in loader:
            with torch.no_grad():
                images = images.to(device, non_blocking=True)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        embed = model(images)
                else:
                    embed = model(images)
            if total_embeddings is None:
                total_embeddings = embed.detach().cpu()
                total_labels = labels.detach().cpu()
            else:
                total_embeddings = torch.cat((total_embeddings, embed.detach().cpu()))
                total_labels = torch.cat((total_labels, labels.detach().cpu()))

            if pbar is not None:
                pbar.update(1)

            del images, labels, embed
    finally:
        if pbar is not None:
            pbar.close()

    #torch.cuda.empty_cache()

    emb = np.float32(total_embeddings)
    lbl = np.uint8(total_labels)
    if is_distributed():
        emb_t = torch.from_numpy(emb).to(device)
        lbl_t = torch.from_numpy(lbl).to(device)
        emb = _all_gather_cat(emb_t).detach().cpu().numpy().astype(np.float32)
        lbl = _all_gather_cat(lbl_t).detach().cpu().numpy().astype(np.uint8)

    return emb, lbl
def train_epoch_constructive(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    ema,
    loss_optimizer,
    scheduler=None,
    scheduler_step_per_batch=False,
    device=torch.device("cuda"),
    grad_accum_steps=1,
    progress_bar=False,
    epoch=None,
):
    model.train()
    train_loss = []
    loss_optimization = loss_optimizer is not None
    grad_accum_steps = max(1, int(grad_accum_steps))
    last_accum = len(train_loader) % grad_accum_steps

    optimizer.zero_grad(set_to_none=True)
    if loss_optimization:
        loss_optimizer.zero_grad(set_to_none=True)

    pbar = None
    if progress_bar and is_main_process():
        epoch_str = f"{epoch + 1}" if epoch is not None else "?"
        pbar = tqdm(total=len(train_loader), desc=f"Train Epoch {epoch_str}", dynamic_ncols=True, leave=False)

    try:
        for idx, (images, labels) in enumerate(train_loader):
            if loss_optimization:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            else:
                images = torch.cat([images[0]["image"], images[1]["image"]], dim=0).to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                bsz = labels.shape[0]

            if scaler is not None:
                with torch.amp.autocast("cuda"):
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

            train_loss.append(loss.item())

            step_now = ((idx + 1) % grad_accum_steps == 0) or ((idx + 1) == len(train_loader))
            accum_denom = grad_accum_steps
            if ((idx + 1) == len(train_loader)) and (last_accum != 0):
                accum_denom = last_accum

            loss_to_backprop = loss / accum_denom

            if scaler is not None:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if step_now:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                    if loss_optimization:
                        scaler.unscale_(loss_optimizer)

                    scaler.step(optimizer)

                    if loss_optimization:
                        scaler.step(loss_optimizer)

                    scaler.update()
                else:
                    optimizer.step()

                    if loss_optimization:
                        loss_optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if loss_optimization:
                    loss_optimizer.zero_grad(set_to_none=True)

            if step_now and scheduler_step_per_batch and scheduler is not None:
                scheduler.step()

            if ema and step_now:
                ema.update(model.parameters())

            if pbar is not None:
                pbar.update(1)
                if step_now:
                    pbar.set_postfix(loss=f"{np.mean(train_loss):.4f}")

            del images, labels, embed, loss, loss_to_backprop
    finally:
        if pbar is not None:
            pbar.close()

    return {"loss": np.mean(train_loss)}

def validation_constructive(
    valid_loader,
    train_loader,
    model,
    device,
    scaler,
    progress_bar=False,
    epoch=None,
    split_name="projection",
):
    """
    This function performs the validation step of the constructive learning algorithm. 

    Parameters:
        valid_loader (torch.utils.data.DataLoader): DataLoader containing the validation data.
        train_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
        model (torch.nn.Module): The model being trained.
        scaler (torch.amp.GradScaler): The scaler used for gradient scaling in case of mixed precision training.

    Returns:
        acc_dict (dict): A dictionary containing the accuracy metrics, computed using the `AccuracyCalculator` class.
    """
    ## capture output containing warnings in buffer

    calculator = AccuracyCalculator(k=1, exclude=["r_precision","mean_average_precision_at_r"])
    model.eval()

    epoch_str = f"{epoch + 1}" if epoch is not None else "?"
    query_embeddings, query_labels = compute_embeddings(
        valid_loader,
        model,
        device,
        scaler,
        progress_bar=progress_bar,
        progress_desc=f"Valid E{epoch_str} ({split_name}) query",
    )
    reference_embeddings, reference_labels = compute_embeddings(
        train_loader,
        model,
        device,
        scaler,
        progress_bar=progress_bar,
        progress_desc=f"Valid E{epoch_str} ({split_name}) ref",
    )
    

    if is_main_process():
        acc_dict = calculator.get_accuracy(
            query_embeddings,
            query_labels,
            reference_embeddings,
            reference_labels,
        )
    else:
        acc_dict = None

    if is_distributed():
        obj = [acc_dict]
        dist.broadcast_object_list(obj, src=0)
        acc_dict = obj[0]

    del query_embeddings, query_labels, reference_embeddings, reference_labels
    #torch.cuda.empty_cache()

    return acc_dict


def train_epoch_ce(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    ema,
    scheduler=None,
    scheduler_step_per_batch=False,
    device=torch.device("cuda"),
    grad_accum_steps=1,
    progress_bar=False,
    epoch=None,
):
    """
    Train the model for one epoch using cross-entropy loss.

    Parameters:
    train_loader (torch.utils.data.DataLoader): The data loader for the training data.
    model (torch.nn.Module): The model to be trained.
    criterion (torch.nn.Module): The loss function to be used for training.
    optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
    scaler (torch.amp.GradScaler): The scaler used for gradient scaling in case of mixed precision training.
    ema (Optional[torch.nn.Module]): The exponential moving average model.

    Returns:
    dict: A dictionary containing the mean loss over the epoch.
    """

    model.train()
    train_loss = []
    grad_accum_steps = max(1, int(grad_accum_steps))
    last_accum = len(train_loader) % grad_accum_steps

    optimizer.zero_grad()

    pbar = None
    if progress_bar and is_main_process():
        epoch_str = f"{epoch + 1}" if epoch is not None else "?"
        pbar = tqdm(total=len(train_loader), desc=f"Train Epoch {epoch_str}", dynamic_ncols=True, leave=False)

    try:
        for batch_i, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            step_now = ((batch_i + 1) % grad_accum_steps == 0) or ((batch_i + 1) == len(train_loader))
            if scaler:
                with torch.amp.autocast("cuda"):
                    output = model(data)
                    loss = criterion(output, target)
                    train_loss.append(loss.item())
                    accum_denom = grad_accum_steps
                    if ((batch_i + 1) == len(train_loader)) and (last_accum != 0):
                        accum_denom = last_accum
                    loss_to_backprop = loss / accum_denom
                    scaler.scale(loss_to_backprop).backward()
                    if step_now:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
            else:
                output = model(data)
                loss = criterion(output, target)
                train_loss.append(loss.item())
                accum_denom = grad_accum_steps
                if ((batch_i + 1) == len(train_loader)) and (last_accum != 0):
                    accum_denom = last_accum
                loss_to_backprop = loss / accum_denom
                loss_to_backprop.backward()
                if step_now:
                    optimizer.step()
                    optimizer.zero_grad()

            if step_now and scheduler_step_per_batch and scheduler is not None:
                scheduler.step()

            if ema and step_now:
                ema.update(model.parameters())

            if pbar is not None:
                pbar.update(1)
                if step_now:
                    pbar.set_postfix(loss=f"{np.mean(train_loss):.4f}")

            del data, target, output
            #torch.cuda.empty_cache()
    finally:
        if pbar is not None:
            pbar.close()

    return {"loss": np.mean(train_loss)}


def validation_ce(model, criterion, valid_loader, device, scaler, progress_bar=False, epoch=None):
    model.eval()
    val_loss = []
    y_pred, y_true = [], []
    correct_samples = 0
    total_samples = 0

    pbar = None
    if progress_bar and is_main_process():
        epoch_str = f"{epoch + 1}" if epoch is not None else "?"
        pbar = tqdm(total=len(valid_loader), desc=f"Valid Epoch {epoch_str}", dynamic_ncols=True, leave=False)

    try:
        for batch_i, (data, target) in enumerate(valid_loader):
            with torch.no_grad():
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        output = model(data)
                        if criterion:
                            loss = criterion(output, target)
                            val_loss.append(loss.item())
                else:
                    output = model(data)
                    if criterion:
                        loss = criterion(output, target)
                        val_loss.append(loss.item())

                target_np = target.detach().cpu().numpy()
                pred_np = np.argmax(output.detach().cpu().numpy(), axis=1)
                correct_samples += (target_np == pred_np).sum()
                total_samples += target_np.shape[0]
                y_pred.append(pred_np)
                y_true.append(target_np)

                if pbar is not None:
                    pbar.update(1)
                    if len(val_loss) > 0:
                        pbar.set_postfix(loss=f"{np.mean(val_loss):.4f}")

                del data, target, output
                #torch.cuda.empty_cache()
    finally:
        if pbar is not None:
            pbar.close()

    y_pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    y_true = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)

    if is_distributed():
        y_pred_t = torch.from_numpy(y_pred).to(device)
        y_true_t = torch.from_numpy(y_true).to(device)
        y_pred = _all_gather_cat(y_pred_t).detach().cpu().numpy()
        y_true = _all_gather_cat(y_true_t).detach().cpu().numpy()

        stats = torch.tensor(
            [sum(val_loss), len(val_loss), float(correct_samples), float(total_samples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum, loss_count, correct_samples, total_samples = stats.tolist()
        valid_loss = (loss_sum / loss_count) if loss_count > 0 else np.nan
    else:
        valid_loss = np.mean(val_loss) if val_loss else np.nan

    f1_scores = f1_score(y_true, y_pred, average=None) if len(y_true) > 0 else np.array([])
    f1_score_macro = f1_score(y_true, y_pred, average='macro') if len(y_true) > 0 else np.nan
    accuracy_score = (correct_samples / total_samples) if total_samples > 0 else np.nan

    metrics = {"loss": valid_loss, "accuracy": accuracy_score, "f1_scores": f1_scores, 'f1_score_macro': f1_score_macro}
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


def save_augmented_sample(data_dir, transform, n_samples, seed):
    """
    Save a sample of augmented images for each class.

    Args:
        data_dir (str): Path to the directory containing the images.
        transform (callable): Transformation to be applied to the images.
        n_samples_per_class (int): Number of images to sample and save per class.
        save_dir (str): Directory to save the augmented image samples.
    """
    # Load dataset
    dataset = ImageFolder(root=os.path.join(data_dir, "train"))
    save_dir = os.path.join(data_dir, "aug_sample")
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    ## reverse image net transforms
    postprocessing = transforms.Compose(
    [
        transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ])

    # Organize samples by class
    class_to_indices = defaultdict(list)
    for idx, (path, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    # Process and save images
    for class_label, indices in class_to_indices.items():
        
        # Randomly select n_samples_per_class indices
        selected_indices = random.sample(indices, min(n_samples, len(indices)))
        class_label_str = dataset.classes[class_label]

        ## apply augmentations and save
        for i, idx in enumerate(selected_indices):
            path, _ = dataset.samples[idx]
            image = Image.open(path).convert("RGB")
            image_name = os.path.basename(path)
            augmented_image = transform(image=np.asarray(image))["image"]   
            to_pil_image = transforms.ToPILImage()
            augmented_image = to_pil_image(postprocessing(augmented_image))
            sample_path = os.path.join(save_dir, f"{class_label_str}_{image_name}_augmented.png")
            augmented_image.save(sample_path)
