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

def collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,image, patch_size, xy_step, z_step):
    # Init indexes
    pz = pz_start
    px = px_start
    py = py_start
    
    ## Collect patchs
    print("collect patchs...")
    patchs = []
    while pz<=pz_end-patch_size[0]:
        while px<=px_end-patch_size[1]:
            while py<=py_end-patch_size[2]: 
                
                ## Slice patch from input       
                px_start_patch = px-px_start
                py_start_patch = py-py_start    
                s = [(pz,pz+patch_size[0]),(px_start_patch,px_start_patch+patch_size[1]),(py_start_patch,py_start_patch+patch_size[2])]
                patch = slice_image(image,s)
                # seg_patch = slice_image(target_seg_image,s)
                
                patchs.append(patch)
                
                py+=min(xy_step,max(1,py_end-patch_size[2]-py))
            py=py_start  
            px+=min(xy_step,max(1,px_end-patch_size[1]-px))
        px=px_start
        pz+=min(z_step,max(1,pz_end-patch_size[0]-pz))
    
    return np.array(patchs)

def assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,patchs,weights,assembled_image_shape,patch_size,xy_step,z_step):
    print("assemble images from patchs...")
    patchs = np.array(patchs)
    assembled_images = np.zeros((patchs.shape[0],*assembled_image_shape))
    pz = pz_start
    px = px_start
    py = py_start 
    i = 0
    while pz<=pz_end - patch_size[0]:
        while px<=px_end-patch_size[1]:
            while py<=py_end-patch_size[2]: 
                px_start_patch = px-px_start
                py_start_patch = py-py_start 
                
                ## Update with patchs
                patch_slice = (slice(pz,pz+patch_size[0]), slice(px_start_patch,px_start_patch+patch_size[1]),slice(py_start_patch,py_start_patch+patch_size[2]))
                
                for j in range(patchs.shape[0]):
                    assembled_images[j][patch_slice] += patchs[j][i]*weights 
                
                py+=min(xy_step,max(1,py_end-patch_size[2]-py))
                i+=1
                
            py=py_start  
            px+=min(xy_step,max(1,px_end-patch_size[1]-px))
        px=px_start
        pz+=min(z_step,max(1,pz_end-patch_size[0]-pz))     
    return assembled_images

##Resize image using TF, this why this code is here and not in ImageUtils
def resize_image(image_ndarray, image):
    # only donwsampling, so use nearest neighbor that is faster to run
    resized_image = np.zeros(image_ndarray)
    for i in range(image.shape[0]):
        resized_image[i] = tf.image.resize(
            image[i], (image_ndarray[1], image_ndarray[2]
                    ), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
    return resized_image