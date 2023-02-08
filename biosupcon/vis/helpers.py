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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def postprocess_image(img):
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
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
    
def scale(arr):
    return ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype("uint8")

def gen_coords(i, patch_size, stride, dim1, dim2):
    x0 = int(stride * (i % dim1))
    y0 = int(stride * int(i / dim2))
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    return x0, y0, x1, y1

def total_variation_regularizer(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(
        torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    )

def normalize_and_scale_features(features, n_sigma=1):
    scaled_features = (features - np.mean(features)) / (np.std(features) )
    scaled_features = np.clip(scaled_features, -n_sigma, n_sigma)
    scaled_features = (scaled_features - scaled_features.min()) / (scaled_features.max()-scaled_features.min())
    return scaled_features

def pca_decomposition(x, n_components=3):
    feats = x.permute(0,2,3,1).reshape(-1, x.shape[1])
    feats = (feats-torch.mean(feats,0))
    u,s,v = torch.svd(feats, compute_uv=True)
    pc = torch.matmul(u[:,:n_components], torch.diag(s[:n_components]))
    pc = pc.view(x.shape[0], x.shape[2], x.shape[3], 3).permute(0,3,1,2)
    return pc

def feature_map_normalization(f):
    act_map = torch.sum(f, dim=1).unsqueeze(1)
    act_map /= act_map.max()
    return act_map

def embbedings_dimension_reductions(data_table):
    mean = np.mean(data_table, axis=0)
    std = np.std(data_table, axis=0)
    norm_data = (data_table - mean) / std
    pca_obj = decomposition.PCA(n_components=2)
    pca = pca_obj.fit_transform(norm_data)
    tsne = manifold.TSNE(learning_rate='auto', init='pca').fit_transform(norm_data)
    names = ['PC1', 'PC2', 'tSNE-0', 'tSNE-1']
    return np.hstack((pca, tsne)), names, pca_obj


def bokeh_plot(df, out_path='plot.html'):
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