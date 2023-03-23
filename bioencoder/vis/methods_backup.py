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
    img_t = preprocess_image(img).to(device)
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

def maximally_activated_patches(model, img, patch_size=448, stride=100, device='cuda', save_path=None):
    """
    Plots the patches of an image that produce the highest activations
    """
    model.eval()
    model.to(device)
    mean = int(np.mean(np.asarray(img)))
    img_t = preprocess_image(img).to(device)

    with torch.no_grad():
        out = model(img_t)

    probs = F.softmax(out[0], dim=0)
    max_index = probs.argmax()
    orig = probs[max_index]

    dim1 = int(((img.shape[0] - patch_size) / stride) + 1)
    dim2 = int(((img.shape[1] - patch_size) / stride) + 1)
    diff = []

    for i in range(dim1 * dim2):
        occluded = img.copy()

        x0, y0, x1, y1 = gen_coords(i, patch_size, stride, dim1, dim2)

        cv2.rectangle(occluded, (x0, y0), (x1, y1), (mean, mean, mean), -1)

        occluded_t = preprocess_image(occluded).to(device)

        with torch.no_grad():
            out = model(occluded_t)

        occ_probs = F.softmax(out[0], dim=0)
        diff.append(abs(orig - occ_probs[max_index].item()))

    diff = np.array(diff.cpu())
    top_indices = diff.argsort()[-5:]
    fig, axs = plt.subplots(int(1), int(5))

    for i, idx in enumerate(top_indices):
        image = img.copy()
        x0, y0, x1, y1 = gen_coords(idx, patch_size, stride, dim1, dim2)
        axs[i].imshow(np.asarray(image[y0:y1, x0:x1]))
        axs[i].set_yticks([])
        axs[i].set_xticks([])

    if save_path:
        fig.savefig(save_path)

def saliency_map(model, img, device = 'cuda', save_path = None):
    """
        Plots the gradient of the score of the predicted class with respect to image pixels
    """

    model.eval()
    model.to(device)    
    img_t = preprocess_image(img).to(device)
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


def generate_image(model, target_class, epochs, min_prob, lr, weight_decay, step_size = 100, gamma = 0.6,
                        noise_size = 224, p_freq = 50, device = 'cuda', save_path = None):
    
    """
        Starting from a random initialization, generates an image that maximizes the score for a specific class using
        gradient ascent
    """
    model.to(device)
    model.eval()

    noise = torch.randn([1, 3, noise_size, noise_size]).to(device)
    noise.requires_grad = True
    opt = torch.optim.SGD([noise], lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma = gamma)
    
    for i in range(1, epochs + 1):
        opt.zero_grad()
        outs = model(noise)
        p = F.softmax(outs[0], dim=0)[target_class]
        
        if i % p_freq == 0 or i == epochs:        
            print('Epoch: {} Confidence score for class {}: {}'.format(i, target_class, p.item()))
            
        if p > min_prob:
            print('Reached {} confidence score in epoch {}. Stopping early.'.format(p.item(), i))
            break
            
        obj = - outs[0][target_class]
        obj.backward()
        opt.step()
        scheduler.step()
    
    fig, axs = plt.subplots(1)
    image = postprocess_image(noise)
    axs.imshow(image)
    axs.set_xticks([])
    axs.set_yticks([])

    if save_path:
        fig.savefig(save_path)
    
    return noise

def fool_model(model, img, target_class, epochs = 500, min_prob =0.9, lr = 0.5, step_size = 100, gamma =0.8,   
                        p_freq = 50, device = 'cuda', save_path = None):
    
    """
        Modifies a given image to have a high score for a specific class, similar to generate_image()
    """

    img_t = preprocess_image(img).to(device)
    img_t.requires_grad = True
    model = model.to(device)
    opt = torch.optim.SGD([img_t], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma = gamma)
    
    for i in range(1, epochs + 1):
        opt.zero_grad()
        outs = model(img_t)
        p = F.softmax(outs[0], dim=0)[target_class]
        
        if i % p_freq == 0 or i == epochs:        
            print('Epoch: {} Confidence score for class {}: {}'.format(i, target_class, p))
            
        if p > min_prob:
            print('Reached {} confidence score in epoch {}. Stopping early.'.format(p, i))
            break
            
        obj = - outs[0][target_class]
        obj.backward()
        opt.step()
        scheduler.step()
    
    fig, axs = plt.subplots(1)
    image = postprocess_image(img_t)
    
    axs.imshow(image)
    axs.set_xticks([])
    axs.set_yticks([])

    if save_path:
        fig.savefig(save_path)
    
    return img_t

def feature_inversion(model, modules, img, epochs, lr, step_size = 100, gamma = 0.6, mu = 1e-1, 
                          device = 'cuda', save_path = None):
    
    """
        Reconstructs an image based on its feature representation at various modules
    """

    img_t = preprocess_image(img).to(device)
    model = model.to(device)
    model.eval()
    recreated_imgs = []
    
    for module in modules:   
        recreated_imgs.append(feature_inversion_helper(model, module, img_t, epochs = epochs,
                                lr = lr, step_size = step_size, gamma = gamma, mu = mu, device = device))
        
    fig, axs = plt.subplots(1, len(recreated_imgs))
    
    for i in range(len(recreated_imgs)):
        axs[i].imshow(postprocess_image(recreated_imgs[i]))


    if save_path:
        fig.savefig(save_path)

def feature_inversion_helper(model, module, img, epochs, lr, step_size, gamma, mu, device = 'cuda'):
    
    """
        Performs feature inversion on one module
    """
    noise_w = img.shape[2]
    noise_h = img.shape[3]
    acts = [0]    
    def hook_fn(self, input, output):
        acts[0] = output
        
    handle = module.register_forward_hook(hook_fn)
    _ = model(img)
    features = acts[0]
       
    noise = torch.randn([1, 3, noise_w, noise_h]).to(device)
    noise.requires_grad = True
    opt = torch.optim.SGD([noise], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma = gamma)
    
    for i in range(epochs):
        opt.zero_grad()
        _ = model(noise)
        curr = acts[0]
        
        loss = ((features - curr) ** 2).sum() + mu * total_variation_regularizer(noise)
        loss.backward(retain_graph = True)
        
        opt.step()
        scheduler.step()
    
    handle.remove()
    return noise

def grad_cam(model, module, img, target_layer = ["4"], target_category= None, device = 'cuda', save_path = None):
    for param in model.parameters():
        param.requires_grad = True
    use_cuda = True if device == 'cuda' else False
        
    grad_cam = GradCam(model = model, feature_module = module,
                    target_layer_names = target_layer, use_cuda = use_cuda)

    img_t = preprocess_image(img).to(device)
        
    grayscale_cam = grad_cam(img_t, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    gb_model = GuidedBackpropReLUModel(model=model.encoder, use_cuda = use_cuda)
    gb = gb_model(img_t, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    fig1 = plt.figure() # create a figure with the default size 

    ax1 = fig1.add_subplot(1,3,1) 
    ax1.imshow(cam[:,:,::-1], interpolation='none')
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

    img_t = preprocess_image(img).to(device)
    
    assert(target_category != None), "Please specify a target category"
    grayscale_cam = contrast_cam(img_t, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    gb_model = GuidedBackpropReLUModel(model=model.encoder, use_cuda = use_cuda)
    gb = gb_model(img_t, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    fig1 = plt.figure() # create a figure with the default size 

    ax1 = fig1.add_subplot(1,3,1) 
    ax1.imshow(cam[:,:,::-1], interpolation='none')
    ax1.set_title('ContrastCam')

    ax2 = fig1.add_subplot(1,3,2)
    ax2.imshow(gb[:,:,::-1], interpolation='none')
    ax2.set_title('Guided Backprop')

    ax3 = fig1.add_subplot(1,3,3)
    ax3.imshow(cam_gb[:,:,::-1], interpolation='none')
    ax3.set_title('GradCam + Guided Backprop')

    
    if save_path:
        fig1.savefig(save_path)
    return fig1


def deep_dream(model, module, img, epochs, lr, step_size = 100, gamma = 0.6, device = 'cuda', save_path = None):

    """
        Modifies the input image to maximize activation at a specific module
    """
    
    img_t = preprocess_image(img).to(device)
    img_t.requires_grad = True
    opt = torch.optim.SGD([img_t], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma = gamma)
    acts = [0]

    def hook_fn(self, input, output):
        acts[0] = output

    model.to(device)
    model.eval()
    handle = module.register_forward_hook(hook_fn)  
    
    for i in range(epochs):
        opt.zero_grad()
        _ = model(img_t)
        loss = -acts[0].norm()
        loss.backward()
        opt.step()
        scheduler.step()
    
    handle.remove()
    
    fig, axs = plt.subplots(1)
    img_np = postprocess_image(img_t)
    
    axs.imshow(img_np)
    axs.set_xticks([])
    axs.set_yticks([])

    if save_path:
        fig.savefig(save_path)
    
    return img_t