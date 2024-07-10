import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from .helpers import *
from .classes import *

plt.rcParams["figure.figsize"] = (16,16)

def visualize_filters(model, filter_name = None, max_filters = 64, size = 128, save_path = None):
    """
        Plots filters of a convolutional layer by interpreting them as grayscale images
    """
    name, weights = next(model.named_parameters())
   
    for layer_name, layer_weights in model.named_parameters():
        if layer_name == filter_name:
            weights = layer_weights 
          
    w_size = weights.size()
    merged_weights = weights.reshape(w_size[0] * w_size[1], w_size[2], w_size[2]).cpu().detach().numpy()
    channels = merged_weights.shape[0]
    
    if channels > max_filters:
        merged_weights = merged_weights[torch.randperm(channels)[:max_filters]]
        channels = max_filters    
    
    sqrt = int(channels**0.5)
    fig, axs = plt.subplots(sqrt, sqrt)
    
    for i in range(sqrt ** 2):
        weight = merged_weights[i]
        scaled = scale(weight)
        resized = transforms.Resize((size, size))(Image.fromarray(scaled))
        idx = int(i / sqrt), i % sqrt
        
        axs[idx].imshow(resized, cmap = 'gray')
        axs[idx].set_yticks([])
        axs[idx].set_xticks([])
    
    if save_path:
        fig.savefig(save_path)
    return fig

def visualize_activations(model, module, img, max_acts = 64, save_path = None, device='cuda'):
    """
        Plots the activations of a module recorded during a forward pass on an image
    """
    model.to(device)
    # img_t = preprocess_image(img).to(device)
    acts = [0]

    def hook_fn(self, input, output):
        acts[0] = output
    
    handle = module.register_forward_hook(hook_fn)
    out = model(img_t)
    handle.remove()
    acts = acts[0][0].cpu().detach().numpy()  # Subset the output for the first copy
    
    if acts.shape[0] > max_acts:
        acts = acts[torch.randperm(acts.shape[0])[:max_acts]]
    
    sqrt = int(acts.shape[0]**0.5)
    fig, axs = plt.subplots(sqrt, sqrt)
    
    for i in range(sqrt ** 2):
        scaled = scale(acts[i])
        idx = int(i / sqrt), i % sqrt
        axs[idx].imshow(scaled, cmap = 'gray')
        axs[idx].set_yticks([])
        axs[idx].set_xticks([])
    
    if save_path:
        fig.savefig(save_path)
    return fig

def saliency_map(model, img, device = 'cuda', save_path = None):
    """
        Plots the gradient of the score of the predicted class with respect to image pixels
    """

    model.eval()
    model.to(device)    
    # img_t = preprocess_image(img).to(device)
    img_t.requires_grad = True
    img_t.retain_grad() #added this line 
    
    out = model(img_t)
    max_out = out[0].max()
    max_out.backward()    
    saliency, _ = torch.max(img_t.grad.data.abs(), dim = 1)
    saliency = saliency.squeeze(0)
    saliency_img = saliency.detach().cpu().numpy()
    
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[1].imshow(saliency_img, cmap = 'gray')
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    
    if save_path:
        plt.savefig(save_path)
    return fig

def grad_cam(model, module, img, target_layer = ["4"], target_category= None, device = 'cuda', save_path = None):
    
    for param in model.parameters():
        param.requires_grad = True

    use_cuda = True if device == 'cuda' else False
        
    grad_cam = GradCam(model = model, feature_module = module,
                    target_layer_names = target_layer, use_cuda = use_cuda)

    # img_t = preprocess_image(img).to(device)
        
    grayscale_cam = grad_cam(img_t, target_category)

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap[:,:,::-1] + np.float32(img)/ 255
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    gb_model = GuidedBackpropReLUModel(model=model.encoder, use_cuda = use_cuda)
    gb = gb_model(img_t, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    fig1 = plt.figure() # create a figure with the default size 

    ax1 = fig1.add_subplot(1,3,1) 
    ax1.imshow(cam, interpolation='none')
    ax1.set_title('GradCam')

    ax2 = fig1.add_subplot(1,3,2)
    ax2.imshow(gb[:,:,::-1], interpolation='none')
    ax2.set_title('Guided Backprop')

    ax3 = fig1.add_subplot(1,3,3)
    ax3.imshow(cam_gb[:,:,::-1], interpolation='none')
    ax3.set_title('GradCam + Guided Backprop')

    
    if save_path:
        fig1.savefig(save_path)
    return fig1


def contrast_cam(model, module, img, target_layer = ["4"], target_category= None, device = 'cuda', save_path = None):
    for param in model.parameters():
        param.requires_grad = True
    use_cuda = True if device == 'cuda' else False
        
    contrast_cam = ContrastCam(model = model, feature_module = module,
                    target_layer_names = target_layer, use_cuda = use_cuda)

    # img_t = preprocess_image(img).to(device)
    
    assert(target_category != None), "Please specify a target category"
    grayscale_cam = contrast_cam(img_t, target_category)

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap[:,:,::-1] + np.float32(img)/ 255
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    gb_model = GuidedBackpropReLUModel(model=model.encoder, use_cuda = use_cuda)
    gb = gb_model(img_t, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    fig1 = plt.figure() # create a figure with the default size 

    ax1 = fig1.add_subplot(1,3,1) 
    ax1.imshow(cam, interpolation='none')
    ax1.set_title('ContrastiveCam')

    ax2 = fig1.add_subplot(1,3,2)
    ax2.imshow(gb[:,:,::-1], interpolation='none')
    ax2.set_title('Guided Backprop')

    ax3 = fig1.add_subplot(1,3,3)
    ax3.imshow(cam_gb[:,:,::-1], interpolation='none')
    ax3.set_title('ContrastiveCam + Guided Backprop')

    
    if save_path:
        fig1.savefig(save_path)
    return fig1
