import streamlit as st
import torch
from PIL import Image
import argparse
from streamlit_option_menu import option_menu

import bioencoder as biocode

scaler = torch.cuda.amp.GradScaler()

# Function to load the model
@st.cache_resource
def load_model(ckpt_pretrained, backbone, num_classes, stage):
    model = biocode.utils.build_model(backbone, second_stage=(stage == 'second'), num_classes=num_classes, ckpt_pretrained=ckpt_pretrained).cuda()
    model.use_projection_head((stage=='second'))
    model.eval()
    return model


def main(args):

    if args.stage == 'first': 
        vis_funcs = ['Filters', 'Activations', 'Saliency']
    else:
        vis_funcs = ['Filters', 'Activations', 'Saliency', 'GradCAM', 'ConstrativeCAM']

    # Set page title and layout
    st.set_page_config(page_title="BioEncoder Model Visualizer", layout="wide")

    # Sidebar
    st.sidebar.image('images/logo.png', use_column_width=True)
    st.sidebar.title("BioEncoder Model Explorer")

    # Image upload
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    # Load the model and add to cache
    model = load_model(args.ckpt_pretrained, args.backbone, args.num_classes, args.stage)

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
            result = biocode.vis.visualize_filters(model.encoder, layer, max_filters=max_filters)
            st.pyplot(result)

        elif selected == 'Activations':
            layers = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.modules.conv.Conv2d)}
            layer = st.selectbox("Select a layer", layers.keys())
            module = layers[layer]
            max_acts = st.slider("Max activations", 5, 64, 25)
            result = biocode.vis.visualize_activations(model, module, image, max_acts=max_acts)
            st.pyplot(result)

        elif selected == 'Saliency':
            result = biocode.vis.saliency_map(model, image)
            st.pyplot(result)

        elif selected == 'GradCAM':
            # add activation type (Relu, Silu, etc_)
            layers =[name.split('.')[0] for name, module in model.encoder.named_modules() if isinstance(module, (torch.nn.SiLU, torch.nn.ReLU))]
            layer_set = sorted(set(layers))
            layer = st.selectbox("Select a layer", list(layer_set), index=len(list(layer_set))-1)
            result = biocode.vis.grad_cam(model, model.encoder,image,target_layer=[layer], target_category= None)
            st.pyplot(result)

        elif selected == 'ConstrativeCAM':
            layers =[name.split('.')[0] for name, module in model.encoder.named_modules() if isinstance(module, (torch.nn.SiLU, torch.nn.ReLU))]
            layer_set = sorted(set(layers))
            layer = st.selectbox("Select a layer", list(layer_set), index=len(list(layer_set))-1)
            target = st.selectbox("Select a target", list(range(args.num_classes)))
            result = biocode.vis.contrast_cam(model, model.encoder,image,target_layer=[layer], target_category= target)
            st.pyplot(result)

if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Model Visualization Playground")
    parser.add_argument('--backbone', type=str, default='timm_tf_efficientnet_b0', help='Path to the pretrained checkpoint')
    parser.add_argument('--ckpt_pretrained', type=str, default=None, help='Path to the pretrained checkpoint')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes for the model')
    parser.add_argument('--stage', type=str, choices=['first', 'second'], default='first', help='Stage of the model (first or second)')

    args, _ = parser.parse_known_args()

    # Run the app
    main(args)