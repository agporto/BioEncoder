#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os 

import streamlit as st
import torch

from streamlit_option_menu import option_menu
from PIL import Image

from bioencoder.core import utils
from bioencoder import vis

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
    

    Parameters
    ----------
    config_path : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ## load bioencoer config
    config = utils.load_config(kwargs.get("bioencoder_config_path"))
    root_dir = config.root_dir
    run_name = config.run_name
    
    class_names = os.listdir(os.path.join(root_dir, "data", run_name, "train"))

    ## load config
    hyperparams = utils.load_yaml(config_path)
    
    ## parse config
    backbone = hyperparams["model"]["backbone"]
    num_classes = hyperparams["model"]["num_classes"]
    stage = hyperparams["model"]["stage"]

    ## get swa path
    ckpt_pretrained = os.path.join(root_dir, "weights", run_name, stage, "swa")
    
    if stage == 'first': 
        vis_funcs = ['Filters', 'Activations', 'Saliency']
    else:
        vis_funcs = ['Filters', 'Activations', 'Saliency', 'GradCAM', 'ConstrativeCAM']

    # Set page title and layout
    st.set_page_config(page_title="BioEncoder Model Visualizer", layout="wide")

    # Sidebar    
    img_path = "https://github.com/agporto/BioEncoder/blob/ae04b88d569b767a688447136cef17dbf4be8c40/images/logo.png?raw=true"
    st.sidebar.image(img_path, use_column_width=True)
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

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.sidebar.image(image, caption="Input Image", use_column_width=True)

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
            result = vis.visualize_activations(model, module, image, max_acts=max_acts)
            st.pyplot(result)

        elif selected == 'Saliency':
            result = vis.saliency_map(model, image)
            st.pyplot(result)

        elif selected == 'GradCAM':
            # add activation type (Relu, Silu, etc_)
            layers =[name.split('.')[0] for name, module in model.encoder.named_modules() if isinstance(module, (torch.nn.SiLU, torch.nn.ReLU))]
            layer_set = sorted(set(layers))
            layer = st.selectbox("Select a layer", list(layer_set), index=len(list(layer_set))-1)
            result = vis.grad_cam(model, model.encoder,image,target_layer=[layer], target_category= None)
            st.pyplot(result)

        elif selected == 'ConstrativeCAM':
            layers =[name.split('.')[0] for name, module in model.encoder.named_modules() if isinstance(module, (torch.nn.SiLU, torch.nn.ReLU))]
            layer_set = sorted(set(layers))
            layer = st.selectbox("Select a layer", list(layer_set), index=len(list(layer_set))-1)
            target = st.selectbox("Select a target", class_names)
            result = vis.contrast_cam(model, model.encoder,image,target_layer=[layer], target_category=class_names.index(target))
            st.pyplot(result)
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
        
    model_explorer(args.config_path)
