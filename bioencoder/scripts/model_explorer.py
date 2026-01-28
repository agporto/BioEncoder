#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os 
import numpy as np
from PIL import Image

import torch
import streamlit as st
from streamlit_option_menu import option_menu
from torchvision.datasets import ImageFolder

from bioencoder import config, utils, vis

#%%

# Function to load the model
@st.cache_resource
def load_model(
        ckpt_pretrained, 
        backbone, 
        num_classes, 
        stage
        ):
    model = utils.build_model(
        backbone, second_stage=(stage == 'second'), 
        num_classes=num_classes, ckpt_pretrained=ckpt_pretrained).cuda()
    model.use_projection_head((stage=='second'))
    model.eval()
    return model


def model_explorer(
        config_path,
        **kwargs,
        ):
    """
    Launches a Streamlit-based web application to interactively explore different visualization
    techniques of the BioEncoder model, such as filter visualizations, activation maps, 
    saliency maps, Grad-CAM, and Contrastive-CAM. It provides a user interface to upload an 
    image, select the visualization type, and dynamically adjust parameters related to each 
    visualization technique.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file that contains model specifications and settings, 
        including backbone, number of classes, and stage-specific settings (e.g., 'first' or 
        'second' stage of model training).

    Raises
    ------
    FileNotFoundError
        If the specified configuration file is not found or required model weights are unavailable.

    Examples
    --------
    To start the model explorer from the command line using a specific configuration:
        bioencoder.model_explorer(config_path=r"bioencoder_configs/explore_stage2.yml")

    """
    
    ## load bioencoer config
    root_dir = config.root_dir
    run_name = config.run_name
        
    ## load config
    hyperparams = utils.load_yaml(config_path)
    
    ## parse config
    backbone = hyperparams["model"]["backbone"]
    num_classes = hyperparams["model"].get("num_classes", None)
    stage = hyperparams["model"]["stage"]
    img_size = hyperparams.get("img_size", None)
    if img_size is None:
        raise ValueError("config must include 'img_size'")
    
    ## get swa path
    ckpt_pretrained = os.path.join(root_dir, "weights", run_name, stage, "swa")
    if stage == 'first': 
        vis_funcs = ['Filters', 'Activations', 'Saliency']
    else:
        vis_funcs = ['Filters', 'Activations', 'Saliency', 'GradCAM', 'ConstrativeCAM']

    # Set page title and layout
    st.set_page_config(page_title="BioEncoder Model Visualizer", layout="wide")

    # Sidebar    
    img_path = "https://github.com/agporto/BioEncoder/raw/main/assets/bioencoder_logo.png"
    st.sidebar.image(img_path, width='stretch')
    st.sidebar.title("BioEncoder Model Explorer")

    # Image upload
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    # Load the model and add to cache
    model = load_model(
        ckpt_pretrained, 
        backbone, 
        num_classes, 
        stage
        )
    
    ## get class names
    train_folder = os.path.join(root_dir, "data", run_name, "train")
    train_folder_timm = ImageFolder(train_folder)
    class_names = train_folder_timm.classes         
                             
    if uploaded_file is not None:
        
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.sidebar.image(image, caption="Input Image", width='stretch')
        
        # resize image
        image_resized = image.resize((img_size, img_size))

        # Generate visualizations
        selected = option_menu(None, vis_funcs, icons=['list' for _ in range(len(vis_funcs))], menu_icon="cast", orientation="horizontal")
        if selected == 'Filters':
            layers = [layer[0] for layer in model.named_parameters() if 'weight' in layer[0]]
            layer = st.selectbox("Select a layer", layers)
            max_filters = st.slider("Max filters", 5, 64, 25)
            result = vis.visualize_filters(model.encoder, layer, max_filters=max_filters)
            st.pyplot(result)

        elif selected == 'Activations':
            layers = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.modules.conv.Conv2d)}
            layer = st.selectbox("Select a layer", layers.keys())
            module = layers[layer]
            max_acts = st.slider("Max activations", 5, 64, 25)
            result = vis.visualize_activations(model, module, image_resized, max_acts=max_acts)
            st.pyplot(result)

        elif selected == 'Saliency':
            result = vis.saliency_map(model, image_resized)
            st.pyplot(result)

        elif selected == 'GradCAM':
            # add activation type (Relu, Silu, etc_)
            layers =[name.split('.')[0] for name, module in model.encoder.named_modules() \
                     if isinstance(module, (torch.nn.SiLU, torch.nn.ReLU))]
            layer_set = sorted(set(layers))
            layer = st.selectbox("Select a layer", list(layer_set), index=len(list(layer_set))-1)
            result = vis.grad_cam(model, model.encoder,image_resized,target_layer=[layer], target_category= None)
            st.pyplot(result)

        elif selected == 'ConstrativeCAM':
            layers =[name.split('.')[0] for name, module in model.encoder.named_modules() \
                     if isinstance(module, (torch.nn.SiLU, torch.nn.ReLU))]
            layer_set = sorted(set(layers))
            layer = st.selectbox("Select a layer", list(layer_set), index=len(list(layer_set))-1)
            target = st.selectbox("Select a target", class_names)
            result = vis.contrast_cam(
                model, model.encoder, image_resized,target_layer=[layer], 
                target_category=class_names.index(target))
            st.pyplot(result)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config-path", type=str, default=None)
    args = parser.parse_args()
            
    model_explorer_cli = utils.restore_config(model_explorer)
    model_explorer_cli(args.config_path)
