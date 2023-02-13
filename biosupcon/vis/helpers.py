import numpy as np
import cv2
import torch
from torchvision import transforms
from sklearn import decomposition, manifold
import matplotlib.pyplot as plt
from bokeh.models import (LassoSelectTool, PanTool,
                          ResetTool,
                          HoverTool, WheelZoomTool)
TOOLS = [LassoSelectTool, PanTool, WheelZoomTool, ResetTool]
from bokeh.models import ColumnDataSource
from bokeh import plotting as bplot


def preprocess_image(img):
    """
    Preprocess the input image for a neural network

    Parameters:
    img (np.ndarray): The input image in numpy format

    Returns:
    torch.Tensor: The preprocessed image in torch tensor format
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def postprocess_image(img):
    """
    Function to post-process an image tensor

    Parameters
    ----------
    img: torch.Tensor
        Input image tensor

    Returns
    -------
    img_np: numpy.ndarray
        The post-processed image in the form of numpy ndarray
    """
    postprocessing = transforms.Compose(
    [
        transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ]
)
    img_post = postprocessing(img[0])
    img_np = img_post.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = scale(np.clip(img_np, 0, 1))
    return img_np

def deprocess_image(img):
    """
    Perform the deprocessing step on an image.

    Parameters:
    -----------
    img : numpy.ndarray
        The image to be deprocessed.

    Returns:
    --------
    deprocessed_img : numpy.ndarray
        The deprocessed image.
    
    References:
    -----------
    https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65
    """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

def show_cam_on_image(img, mask):
    """
    Combine the provided heatmap (mask) with the original image to highlight the regions with higher activations.
    
    Parameters:
    img (np.uint8): Original image as a numpy array of shape (H, W, 3) with values in [0, 255].
    mask (np.uint8): Heatmap as a numpy array of shape (H, W) with values in [0, 1].
    
    Returns:
    np.uint8: Combined image of shape (H, W, 3) with values in [0, 255].
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
    
def scale(arr):
    """
    Scale the values of a given numpy array `arr` to the range [0, 255].
    
    Parameters:
    ----------
    arr : numpy.ndarray
        The input numpy array.
    
    Returns:
    -------
    numpy.ndarray
        The rescaled numpy array with values in the range [0, 255].
    """

    return ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype("uint8")

def gen_coords(i, patch_size, stride, dim1, dim2):
    """
    Generates the coordinates for a patch given an index, patch size, stride, and image dimensions.

    Parameters
    ----------
    i : int
        Index of the patch.
    patch_size : int
        Size of the patch in pixels.
    stride : int
        Stride size in pixels.
    dim1 : int
        First dimension of the image.
    dim2 : int
        Second dimension of the image.

    Returns
    -------
    tuple
        Tuple containing the (x0, y0, x1, y1) coordinates of the patch.
    """
    x0 = int(stride * (i % dim1))
    y0 = int(stride * int(i / dim2))
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    return x0, y0, x1, y1

def total_variation_regularizer(x):
    """
    This function calculates the total variation of a 4D tensor x by summing up the absolute differences between consecutive elements along the last two dimensions.

    Args:
    x (torch.Tensor): 4D tensor to calculate the total variation for.

    Returns:
    torch.Tensor: The total variation of the input tensor.

    """
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(
        torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    )

def normalize_and_scale_features(features, n_sigma=1):
    """
    Normalizes and scales the input features.

    Parameters:
    features (numpy array): An array of numerical features to be normalized and scaled.
    n_sigma (int, optional): The number of standard deviations used for clipping. The default is 1.

    Returns:
    numpy array: A normalized and scaled array of features.

    """
    scaled_features = (features - np.mean(features)) / (np.std(features) )
    scaled_features = np.clip(scaled_features, -n_sigma, n_sigma)
    scaled_features = (scaled_features - scaled_features.min()) / (scaled_features.max()-scaled_features.min())
    return scaled_features

def pca_decomposition(x, n_components=3):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Parameters:
    x (torch tensor): A 4-dimensional tensor of shape (batch_size, num_features, height, width).
    n_components (int, optional): Number of principal components to keep. The default is 3.

    Returns:
    torch tensor: A 4-dimensional tensor of shape (batch_size, n_components, height, width) 
                  that contains the PCA components.

    """
    feats = x.permute(0,2,3,1).reshape(-1, x.shape[1])
    feats = (feats-torch.mean(feats,0))
    u,s,v = torch.svd(feats, compute_uv=True)
    pc = torch.matmul(u[:,:n_components], torch.diag(s[:n_components]))
    pc = pc.view(x.shape[0], x.shape[2], x.shape[3], 3).permute(0,3,1,2)
    return pc

def feature_map_normalization(f):
    """
    Normalize the feature map.

    Parameters:
    f (torch tensor): A feature map of shape (batch_size, num_features, height, width).

    Returns:
    torch tensor: A normalized feature map of shape (batch_size, 1, height, width).

    """
    act_map = torch.sum(f, dim=1).unsqueeze(1)
    act_map /= act_map.max()
    return act_map

def embbedings_dimension_reductions(data_table):
    """
    Perform dimension reduction on the input data.

    Parameters:
    data_table (numpy array): A 2-dimensional array of shape (num_samples, num_features).

    Returns:
    numpy array, list of str, object: A 2-dimensional array of shape (num_samples, 4) that contains 
                                      the dimension reduced data, a list of column names, and the PCA object.

    """
    mean = np.mean(data_table, axis=0)
    std = np.std(data_table, axis=0)
    norm_data = (data_table - mean) / std
    pca_obj = decomposition.PCA(n_components=2)
    pca = pca_obj.fit_transform(norm_data)
    tsne = manifold.TSNE(learning_rate='auto', init='pca').fit_transform(norm_data)
    names = ['PC1', 'PC2', 'tSNE-0', 'tSNE-1']
    return np.hstack((pca, tsne)), names, pca_obj


def bokeh_plot(df, out_path='plot.html'):
    """
    Plot a scatter plot of the PCA and t-SNE dimensions of the data using bokeh.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to plot. The dataframe must have
                           two columns: 'paths' (the file paths of the images) and 'class' (the
                           class labels of the images).
    out_path (str, optional): The file path to save the bokeh plot to. Default is 'plot.html'.

    Returns:
    bokah.plotting.Figure: The bokeh figure object.

    Raises:
    ValueError: If the dataframe does not have columns 'paths' and 'class'.
    """
    
    if not all(col in df.columns for col in ['paths', 'class']):
        raise ValueError("The dataframe must have columns 'paths' and 'class'")
        
    tooltip = """
        <div>
            <div>
                <img
                src="@image_files" height="192" alt="image"
                style="float: left; margin: 0px 15px 15px 0px; image-rendering: pixelated;"
                border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px;">@class_str</span>
            </div>
        </div>
              """
    filenames = df['paths']
    df['image_files'] = filenames
    num_classes = len(df['class'].unique())
    cmap=plt.cm.get_cmap("jet", num_classes)
    colors_raw = cmap((df['class']), bytes=True)
    colors_str = ['#%02x%02x%02x' % tuple(c[:3]) for c in colors_raw]
    df['color'] = colors_str
    source = ColumnDataSource(df)
    bplot.output_file(out_path)
    hover0 = HoverTool(tooltips=tooltip)
    hover1 = HoverTool(tooltips=tooltip)
    tools0 = [t() for t in TOOLS] + [hover0]
    tools1 = [t() for t in TOOLS] + [hover1]
    pca = bplot.figure(tools=tools0)
    pca.circle('PC1', 'PC2', color='color', source=source)
    tsne = bplot.figure(tools=tools1)
    tsne.circle('tSNE-0', 'tSNE-1', color='color', source=source)
    p = bplot.gridplot([[pca, tsne]])
    bplot.show(p)
    return p