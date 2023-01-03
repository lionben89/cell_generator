from cmath import nan
from cell_imaging_utils.image.image_utils import ImageUtils
import numpy as np
import os
import math 

def slice_image(image_ndarray: np.ndarray, indexes: list) -> np.ndarray:
    n_dim = len(image_ndarray.shape)
    slices = [slice(None)] * n_dim
    for i in range(len(indexes)):
        if indexes[i] is None:
            slices[i] = slice(None)
        else:
            slices[i] = slice(indexes[i][0], indexes[i][1])
    slices = tuple(slices)
    sliced_image = image_ndarray[slices]
    return sliced_image

def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def preprocess_image(dataset,image_index,channels,normalize=True,slice_by=None):
    images = []
    image_path = dataset.df.get_item(image_index,'path_tiff')
    image_ndarray = ImageUtils.image_to_ndarray(ImageUtils.imread(image_path))
    for i in range(len(channels)):
        channel = channels[i]
        channel_index = dataset.df.get_item(image_index,channel)
        image = None
        if channel_index is not None and not math.isnan(channel_index):
            channel_index = int(channel_index)
            image = ImageUtils.get_channel(image_ndarray,channel_index)
            image = np.expand_dims(image[0], axis=-1)
            if normalize == True or normalize[i]:
                max_var = np.max(image!=np.inf)
                image = np.where(image==np.inf,max_var,image)
                mean = np.mean(image,dtype=np.float64)
                std = np.std(image,dtype=np.float64)
                image = (image-mean)/std
            if slice_by is not None:
                image = slice_image(image,slice_by)
            if image.shape[0] < 32:
                image = np.pad(image,[(math.ceil((32-image.shape[0])/2),math.ceil((32-image.shape[0])/2)),(0,0),(0,0),(0,0)],'edge')
        images.append(image)
    return images