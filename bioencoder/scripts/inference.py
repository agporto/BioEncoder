#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% imports


import os
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm 

from bioencoder import config
from bioencoder.core import utils

#%% function

def inference(    
        config_path, 
        image,
        **kwargs,
):

    """
    Generates embeddings or class predictions for a given image using a trained model.
    This function loads the model configuration, processes the input image, and computes
    the embeddings or class probabilities based on the specified model stage.

    Parameters
    ----------
    config_path : str
        Path to the YAML file that contains settings for the model.
        This configuration includes details on model architecture, data loaders, and other
        hyperparameters required for embedding computation.
    image : str or np.ndarray
        Path to the image file or an image represented as a numpy array.
        The image is processed and transformed before being passed through the model.

    Returns
    -------
    np.ndarray or dict
        If stage is 'first', returns the image embeddings as a numpy array.
        If stage is 'second' and return_probs is False, returns the class with the highest probability.
        If stage is 'second' and return_probs is True, returns a dictionary with class probabilities.
   
    Examples
    --------
    To generate embeddings for a given image:
        embeddings = bioencoder.inference("/path/to/config.yaml", "/path/to/image.jpg")

    """
        
    ## load bioencoer config
    run_name = config.run_name
    root_dir = config.root_dir

    ## load config
    hyperparams = utils.load_yaml(config_path)
    
    ## parse config
    backbone = hyperparams["model"]["backbone"]
    num_classes = hyperparams["model"].get("num_classes", None)
    checkpoint = hyperparams["model"].get("checkpoint", "swa")
    checkpoint_path = hyperparams["model"].get("checkpoint_path", None)
    stage = hyperparams["model"].get("stage", "first")
    return_probs = hyperparams.get("return_probs", False)
    device = hyperparams.get("device", "cuda")

    ## load weights
    if checkpoint_path:
        ckpt_pretrained = checkpoint_path
    else:    
        ckpt_pretrained = os.path.join(config.root_dir, "weights", run_name, stage, checkpoint)
        
    ## load from config
    if root_dir and run_name:
        train_dir = os.path.join(root_dir,"data",  run_name, "train")
        labels_sorted = ImageFolder(root=train_dir).classes
        
    ## set random seed
    utils.set_seed()

    ## get transformations
    transform = utils.get_transforms(hyperparams, no_aug=True)

    ## build model
    if config.model_path != ckpt_pretrained:
        print(f"loading checkpoint: {ckpt_pretrained}")
        model = utils.build_model(
            backbone,
            second_stage=(stage == "second"),
            num_classes=num_classes,
            ckpt_pretrained=ckpt_pretrained,
        )
        model.to(device)
        config.model = model
        config.model_path = ckpt_pretrained
    else: 
        model = config.model
        
    ## set to eval
    model.eval()
         
    ## load file
    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"File does not exist: {image}")
        image = np.asarray(Image.open(image))
    elif not isinstance(image, (np.ndarray, np.generic)):
        raise TypeError("Input must be either an image path (str) or a NumPy array.")
    
    ## transform image and move to GPU
    image = transform(image=image)["image"]
    image = image.unsqueeze(0).to(device)

    ## get embeddings / logits
    with torch.no_grad():
        output = model(image)
    
    if stage=="first":
        model.use_projection_head(False)
        embeddings = output.detach().cpu().numpy().squeeze()  
        embeddings = np.float32(embeddings)
        result = embeddings    
        
    elif stage=="second":
        model.use_projection_head((stage=='second'))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probabilities = probabilities.detach().cpu().numpy().squeeze()  
        classes = {class_name: float(prob) for class_name, prob in zip(labels_sorted, np.float32(probabilities))}
        if return_probs:
            result = classes
        else:
            result = max(classes, key=classes.get)
        
    return result

def cli():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",type=str, help="Path to the YAML configuration file to create interactive plots.")
    parser.add_argument("--path", type=str, help="Path to image or folder to embedd / classify.")
    parser.add_argument("--save-path", type=str, default="bioencoder_results.csv", help="Path to CSV file with results.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite CSV file with results.")
    args = parser.parse_args()
    
    inference_cli = utils.restore_config(inference)
    
    if os.path.isdir(args.path):
        if os.path.isfile(args.save_path) and not args.overwrite:
            return f"{args.save_path} exists - (use --overwrite)"
        file_name_list = os.listdir(args.path)
        results_dict = {}
        for file_name in tqdm(file_name_list, desc="Processing files"):
            file_path = os.path.join(args.path, file_name)
            results_dict[file_name] = inference_cli(args.config_path, image=file_path)
        
        data_results = pd.DataFrame.from_dict(results_dict, orient="index")
        if len(list(data_results))==1:
            data_results.rename(columns={0: "class"}, inplace=True)
        data_results.reset_index(inplace=True)
        data_results.rename(columns={"index": "image_name"}, inplace=True)
        data_results.to_csv(args.save_path, index=False)
        print(f"saved BioEncoder results to {args.save_path}")
    elif os.path.isfile(args.path):
        result = {os.path.basename(args.path): inference_cli(args.config_path, image=args.path)}
        return result
        
if __name__ == "__main__":
    
    cli()
