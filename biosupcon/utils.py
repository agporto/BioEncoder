import torch
import random
import os
import numpy as np
from sklearn.metrics import f1_score

from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from .losses import LOSSES
from .optimizers import OPTIMIZERS
from .schedulers import SCHEDULERS
from .models import SupConModel
from .datasets import create_dataset
from .augmentations import get_transforms


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_to_logs(logging, message):
    logging.info(message)


def add_to_tensorboard_logs(writer, message, tag, index):
    writer.add_scalar(tag, message, index)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, crop_transform):
        self.crop_transform = crop_transform

    def __call__(self, x):
        return [self.crop_transform(image=x), self.crop_transform(image=x)]


def build_transforms(config):
    train_transforms = get_transforms(config)
    valid_transforms = get_transforms(config, valid=True)

    return {
        "train_transforms": train_transforms,
        "valid_transforms": valid_transforms
    }


def build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=False):

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
            transform=TwoCropTransform(transforms['train_transforms']), 
            second_stage=False
        )

        train_supcon_loader = torch.utils.data.DataLoader(
            train_supcon_dataset, 
            batch_size=batch_sizes['train_batch_size'], 
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=True
        )

        loaders['train_supcon_loader'] = train_supcon_loader

    else:
        return loaders
    


def build_model(backbone, second_stage=False, num_classes=None, ckpt_pretrained=None):
    model = SupConModel(backbone=backbone, second_stage=second_stage, num_classes=num_classes)

    if ckpt_pretrained:
        model.load_state_dict(torch.load(ckpt_pretrained)['model_state_dict'], strict=False)

    return model


def build_optim(model, optimizer_params, scheduler_params, loss_params):
    if 'params' in loss_params:
        criterion = LOSSES[loss_params['name']](**loss_params['params'])
    else:
        criterion = LOSSES[loss_params['name']]()

    optimizer = OPTIMIZERS[optimizer_params["name"]](model.parameters(), **optimizer_params["params"])

    if scheduler_params:
        scheduler = SCHEDULERS[scheduler_params["name"]](optimizer, **scheduler_params["params"])
    else:
        scheduler = None

    return {"criterion": criterion, "optimizer": optimizer, "scheduler": scheduler}


def compute_embeddings(loader, model, scaler):
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    total_embeddings = np.zeros((len(loader)*loader.batch_size, model.embed_dim))
    total_labels = np.zeros(len(loader)*loader.batch_size)

    for idx, (images, labels) in enumerate(loader):
        images = images.cuda()
        bsz = labels.shape[0]
        if scaler:
            with torch.cuda.amp.autocast():
                embed = model(images)
                total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
                total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()
        else:
            embed = model(images)
            total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
            total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()

        del images, labels, embed
        torch.cuda.empty_cache()

    return np.float32(total_embeddings), total_labels.astype(int)


def train_epoch_constructive(train_loader, model, criterion, optimizer, scaler, ema):
    model.train()
    train_loss = []

    for idx, (images, labels) in enumerate(train_loader):
        images = torch.cat([images[0]['image'], images[1]['image']], dim=0)
        images = images.cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        if scaler:
            with torch.cuda.amp.autocast():
                embed = model(images)
                f1, f2 = torch.split(embed, [bsz, bsz], dim=0)
                embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = criterion(embed, labels)

        else:
            embed = model(images)
            f1, f2 = torch.split(embed, [bsz, bsz], dim=0)
            embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(embed, labels)

        del images, labels, embed
        torch.cuda.empty_cache()

        train_loss.append(loss.item())

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema:
            ema.update(model.parameters())

    return {'loss': np.mean(train_loss)}


def validation_constructive(valid_loader, train_loader, model, scaler):
    calculator = AccuracyCalculator(k=1)
    model.eval()

    query_embeddings, query_labels = compute_embeddings(valid_loader, model, scaler)
    reference_embeddings, reference_labels = compute_embeddings(train_loader, model, scaler)

    acc_dict = calculator.get_accuracy(
        query_embeddings,
        reference_embeddings,
        query_labels,
        reference_labels,
        embeddings_come_from_same_source=False
    )

    del query_embeddings, query_labels, reference_embeddings, reference_labels
    torch.cuda.empty_cache()

    return acc_dict


def train_epoch_ce(train_loader, model, criterion, optimizer, scaler, ema):
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
    model.eval()
    val_loss = []
    valid_bs = valid_loader.batch_size
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    y_pred, y_true = np.zeros(len(valid_loader)*valid_bs), np.zeros(len(valid_loader)*valid_bs)
    correct_samples = 0

    for batch_i, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    if criterion:
                        loss = criterion(output, target)
                        val_loss.append(loss.item())
            else:
                output = model(data)
                if criterion:
                    loss = criterion(output, target)
                    val_loss.append(loss.item())

            correct_samples += (
                target.detach().cpu().numpy() == np.argmax(output.detach().cpu().numpy(), axis=1)
            ).sum()
            y_pred[batch_i * valid_bs : (batch_i + 1) * valid_bs] = np.argmax(output.detach().cpu().numpy(), axis=1)
            y_true[batch_i * valid_bs : (batch_i + 1) * valid_bs] = target.detach().cpu().numpy()

            del data, target, output
            torch.cuda.empty_cache()

    valid_loss = np.mean(val_loss)
    f1_scores = f1_score(y_true, y_pred, average=None)
    f1_score_macro = f1_score(y_true, y_pred, average='macro')
    accuracy_score = correct_samples / (len(valid_loader)*valid_bs)

    metrics = {"loss": valid_loss, "accuracy": accuracy_score, "f1_scores": f1_scores, 'f1_score_macro': f1_score_macro}
    return metrics


def copy_parameters_from_model(model):
    copy_of_model_parameters = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    return copy_of_model_parameters


def copy_parameters_to_model(copy_of_model_parameters, model):
    for s_param, param in zip(copy_of_model_parameters, model.parameters()):
        if param.requires_grad:
            param.data.copy_(s_param.data)
