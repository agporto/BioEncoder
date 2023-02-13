import cv2
import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Function, Variable

#plt.rcParams["figure.figsize"] = (20,20)

class FeatureExtractor:
    """
    Class for extracting activations and registering gradients from targeted intermediate layers of a model.
    
    Args:
        model (nn.Module): The model from which the activations and gradients will be extracted.
        target_layers (list): List of strings containing the names of the target layers in the model.
    
    Methods:
        save_gradient(grad): Method to store the gradients of the target layers.
        __call__(x): Call operator to extract the activations and gradients from the target layers.
                      Returns the activations and the final output of the model.
    """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs:
    """
    Class for getting the network output, activations, and gradients 
    from intermediate targetted layers after a forward pass.
    
    Args:
        model (nn.Module): The model that will be used for the forward pass.
        feature_module (nn.Module): The module that contains the target layers.
        target_layers (list): List of strings containing the names of the target layers in the model.
    
    Methods:
        get_gradients(): Method to return the gradients of the target layers.
        __call__(x): Call operator to make a forward pass through the model and get the output, 
                      activations, and gradients from the target layers. Returns the activations and output of 
                      the model.
    """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x

class GradCam:
    """
    Class for generating Grad-CAM visualizations for a given target layer in a given model.

    Args:
        model (nn.Module): The model to generate Grad-CAM visualizations for.
        feature_module (nn.Module): The feature extraction module within the model to extract activations from.
        target_layer_names (list of str): The names of the target layers to extract activations from.
        use_cuda (bool): Flag indicating whether to use CUDA tensors or not.
    
    """
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    """
    Class for implementing the GuidedBackpropReLU activation function in PyTorch.
    This function is used for guided backpropagation in the network.

    Attributes:
        forward (staticmethod): the forward pass method.
        backward (staticmethod): the backward pass method.
    """
    @staticmethod
    def forward(self, input_img):
        """
        The forward pass method of the activation function.

        Args:
            input_img (torch.Tensor): input image tensor.

        Returns:
            torch.Tensor: the output tensor.
        """
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        """
        The backward pass method of the activation function.

        Args:
            grad_output (torch.Tensor): the gradient tensor from the previous layer.

        Returns:
            torch.Tensor: the gradient tensor for the input.
        """
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    """
    A class that creates a model with GuidedBackpropReLU activation functions instead of standard ReLU activations.
    This class replaces ReLU activations in a given model with GuidedBackpropReLU activations, making it easier to visualize
    the importance of each neuron in the model's predictions.

    Args:
    model (nn.Module): The original model whose activations are to be replaced.
    use_cuda (bool): A flag indicating whether to use GPU or not.
    """

    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

class ContrastCam:
    """
    ContrastCam - Implementation of a contrastive Grad-CAM for a given model and target layer.

    Attributes:
    model (torch.nn.Module): The model to be used.
    feature_module (torch.nn.Module): The feature module used to extract activations from the target layer.
    target_layer_names (list of str): A list of target layer names.
    cuda (bool): A flag indicating whether CUDA should be used.

    Methods:
    forward (input_img): Passes an input image through the model.
    call (input_img, target_category=None): Generates a contrastive Grad-CAM for a given input image and target category.

    """
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)
        
    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        #Change from GradCam
        ce_loss = nn.CrossEntropyLoss()
        im_label_as_var = Variable(torch.from_numpy(np.asarray([target_category])))
        pred_loss = ce_loss(output.cuda(), im_label_as_var.cuda()) 


        self.feature_module.zero_grad()
        self.model.zero_grad()
        pred_loss.backward()

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam