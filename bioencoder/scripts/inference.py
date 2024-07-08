import os
import argparse
import numpy as np
import pickle
import torch
from torchvision.datasets import ImageFolder
from PIL import Image

from bioencoder import config
from bioencoder.core import utils

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
    standardize = hyperparams.get("standardize", False)
    return_probs = hyperparams.get("return_probs", False)

    ## load weights
    if checkpoint_path:
        ckpt_pretrained = checkpoint_path
    else:    
        ckpt_pretrained = os.path.join(config.root_dir, "weights", run_name, stage, checkpoint)

    ## set random seed
    utils.set_seed()

    ## get transformations
    transform = utils.get_transforms(hyperparams, valid=False)

    ## build model
    if config.model_path != ckpt_pretrained:
        print(f"loading checkpoint: {ckpt_pretrained}")
        model = utils.build_model(
            backbone,
            second_stage=(stage == "second"),
            num_classes=num_classes,
            ckpt_pretrained=ckpt_pretrained,
        ).cuda()
        config.model = model
        config.model_path = ckpt_pretrained
    else: 
        model = config.model
        
    ## set to eval
    model.eval()

    ## get labels
    train_dir = os.path.join(root_dir,"data",  run_name, "train")
    labels_sorted = ImageFolder(root=train_dir).classes
         
    
    if isinstance(image, str):
        if os.path.isfile(image):
            image = Image.open(image)
            image = np.asarray(image)
        else:
            print("File does not exist")
            return
    elif isinstance(image, (np.ndarray, np.generic)):
        print("image shape:" + str(image.shape))
        # Input is already a numpy array or an instance of np.generic (which np.ndarray inherits from)
        pass
    else:
        print("Wrong format - need either image path or array type")
        return
    
    ## transform image and move to GPU
    image = transform(image=image)["image"]
    image = image.unsqueeze(0).cuda()
        
    ## get embeddings / logits
    with torch.no_grad():
        output = model(image)
    
    if stage=="first":
        model.use_projection_head(False)
        embeddings = output.detach().cpu().numpy().squeeze()  
        embeddings = np.float32(embeddings)
        if standardize:
            mean = np.mean(embeddings, axis=0)
            std = np.std(embeddings, axis=0)
            embeddings = (embeddings - mean) / std
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
    parser.add_argument("--image", type=str, help="Path to image to embedd / classify.")
    args = parser.parse_args()
    
    inference_cli = utils.restore_config(inference)
    result = inference_cli(args.config_path, image=args.image)
    print(result)
    
if __name__ == "__main__":
    
    cli()
