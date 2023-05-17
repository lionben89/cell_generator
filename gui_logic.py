import gc
import tensorflow as tf
from numpy import dtype
from scipy import signal
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
from metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from metrics import *
from dataset import DataGen
import global_vars as gv
from utils import *
import os
import matplotlib.pyplot as plt
import cv2
from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

root_dir = "/sise/home/lionb/"

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def run_modules(self, module, name, outputs, x, skip_x):
        if ((name == 'sub_2conv_less') or (name == 'sub_2conv_more')
                or (not module._modules)):
            # primitive layer

            if name == 'sub_2conv_less':
                x = torch.cat((skip_x[-2], x), 1)  # concatenate
                skip_x = skip_x[:-2] + skip_x[-1:]
            x = module(x)
            if name == 'sub_2conv_more':
                skip_x += [x]
            if module in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

            return outputs, x, skip_x

        for name, module in module._modules.items():
            outputs, x, skip_x = self.run_modules(module, name, outputs, x,
                                                  skip_x)
        return outputs, x, skip_x

    def __call__(self, x):
        outputs = []
        skip_x = []
        self.gradients = []
        outputs, x, skip_x = self.run_modules(self.model, 'net', outputs, x,
                                              skip_x)
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module,
                                                  target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        target_activations, x = self.feature_extractor(x)
        # x=self.model.out(x)

        return target_activations, x

class PixelRoi():
    def __init__(self, i, j, k, signal):
        self.roi = np.zeros(signal.shape)
        self.roi[k, j, i] = 1

class FullRoi():
    def __init__(self, k, signal):
        self.roi = np.ones_like(signal)

class SubsetRoi():
    def __init__(self, i, j, i2, j2, k, signal):
        self.roi = np.zeros(signal.shape)
        self.roi[k, j:j2, i:i2] = 1

def add_cam_image(img, mask, i):
    alpha = 0.15
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = np.float32(np.dstack([img, img, img]))
    cv2.addWeighted(heatmap, alpha, cam, 1 - alpha, 0, cam)
    cam = cam / np.max(cam)
    ImageUtils.imsave(np.uint8(255 * cam),
                      "{}/{}_{}.tiff".format(root_dir, "test", i))

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model.cuda()
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def __call__(self, input, roi):
        with tf.GradientTape() as tape:
            pred = tf.sum(self.model(input, training=False)*roi)
            class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
            loss = pred[0][class_idxs_sorted[0]]
            
        grads = tape.gradient(loss, input)
        dgrad_abs = tf.math.abs(grads)

        dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

        ## normalize to range between 0 and 1
        arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
        grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
        return grad_eval

class Saliency:
    def __init__(self,model):
        self.model = model
    
    def __call__(self, input, roi):
        if np.sum(roi)>0:
            input_tensor = tf.Variable(np.expand_dims(input,axis=0), dtype=tf.float16)
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                pred = self.model(input_tensor, training=False)*roi
                loss = tf.reduce_sum(pred)
                            
            grads = tape.gradient(loss, input_tensor)
            dgrad_abs = tf.math.abs(grads)


            ## normalize to range between 0 and 1
            arr_min, arr_max  = np.min(dgrad_abs), np.max(dgrad_abs)
            grad_eval = (dgrad_abs - arr_min) / (arr_max - arr_min + 1e-18)
            return grad_eval.numpy()[0]
        else:
            return roi
        
class GradCam:
    def __init__(self,
                 model,
                 feature_module,
                 target_layer_names,
                 use_cuda,
                 th=0,
                 X_gradcam=False,
                 pos_class=True):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.th = th
        self.X_gradcam = X_gradcam
        self.pos_class = pos_class

        self.extractor = ModelOutputs(self.model, self.feature_module,
                                      target_layer_names)

    def forward(self, input):
        return self.model(input)

    def get_output_and_features(self, input):

        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        return features, output

    def __call__(self, input, index, roi, is_segmentation=False):
        if self.cuda:
            features, output = self.extractor(input.cuda())
            #print(features[-1].shape)
            #print(output.shape)# [1, 2048, 7, 7], [1, 1000]
        else:
            features, output = self.extractor(input)
        
        ##for segmentation
        if is_segmentation:
            output = (torch.sigmoid(output)) 
            if (self.pos_class):
                output = torch.where(output >= self.th, output,
                                    torch.full_like(output, 0))
            else:
                output = torch.where(output < self.th, output,
                                    torch.full_like(output, 0))
            output = torch.where(output == self.th, torch.full_like(output, 1), torch.full_like(output, 0)) ## why 0 like in ER
        
        if self.cuda:
            # one_hot = torch.sum(output)
            one_hot = torch.sum(output[0] * torch.from_numpy(roi).cuda())
            #print(one_hot)
        else:
            # one_hot = torch.sum( output)
            one_hot = torch.sum(output[0] * torch.from_numpy(roi))
            

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy(
        )  # 1, 2048, 7, 7
        target = features[-1]
        #print(target.shape)# 1,2048,7,7
        target = target.cpu().data.numpy()[0, :]  # 2048, 7, 7

        if (not self.X_gradcam):
            weights = np.mean(grads_val,
                              axis=(2, 3,
                                    4))[0, :]  # 2048  Gradcam implementation
        else:
            weights = np.sum(grads_val[0, :] * target,
                             axis=(1, 2, 3))  # 2048  X-Gradcam implementation
            weights = weights / (np.sum(target, axis=(1, 2, 3)) + 1e-6
                                 )  # X-Gradcam continue

        # weights = np.maximum(weights, 0)
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):  # w:weight,target:feature
            cam += w * target[i, :, :, :]

        cam = np.maximum(cam, 0)  # 7,7
        # cam = cv2.resize(cam, input.shape[2:])  # 224,224
        ## resizze
        width = input.shape[3]
        height = input.shape[4]
        depth = input.shape[2]
        resized_cam = torch.nn.functional.interpolate(
            torch.from_numpy(cam).unsqueeze(0).unsqueeze(0),
            size=(depth, width, height),
            mode='trilinear').numpy()
        resized_cam = resized_cam[0, :, :, :, :]
        resized_cam = resized_cam - np.min(resized_cam)
        resized_cam = resized_cam / (np.max(resized_cam) + 0.0001)
        return resized_cam

def load_model(model_path):
    ## Load model
    print("Loading model:",model_path)
    model = keras.models.load_model(model_path)
    return model

def get_dataset(ds_path):
    dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
    return dataset

def item_from_dataset(dataset,image_index,slice_by=None):
    ## input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image
    input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image =  preprocess_image(dataset,int(image_index),[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane","membrane_seg"],normalize=[True,True,False,True,True,False],slice_by=slice_by)
    if target_seg_image is not None:
        target_seg_image = target_seg_image/255.
    else:
        target_seg_image = np.zeros_like(target_image)
        
    if mem_seg_image is not None:
        # mem_seg_image = np.ones_like(target_image)
        mem_seg_image = mem_seg_image/255.
    else:
        mem_seg_image = np.ones_like(target_image)
    return input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image
    
def _get_weights(shape):
    shape_in = shape
    shape = shape[1:]
    weights = 1
    for idx_d in range(len(shape)):
        slicey = [np.newaxis] * len(shape)
        slicey[idx_d] = slice(None)
        size = shape[idx_d]
        values = signal.triang(size)
        weights = weights * values[tuple(slicey)]
    return np.broadcast_to(weights, shape_in).astype(np.float32)

def get_mask(interperter,
             signal,
             roi_mode="full",
             roi_args=None,
             is_save=True,
            ):
    
    tf.keras.backend.clear_session()
    _ = gc.collect() 
    px_start = 0
    py_start = 0
    pz_start = 0
    px_end = signal.shape[1]
    py_end = signal.shape[2]
    pz_end = signal.shape[0]
    xy_step = 64
    z_step = 16
    roi = None
    
    # roi = PixelRoi(122,146,14,signal)
    if roi_mode == "full":
        roi = FullRoi(*roi_args, signal)
    elif roi_mode == "pixel":
        roi = PixelRoi(*roi_args, signal)
    elif roi_mode == "subset":
        roi = SubsetRoi(*roi_args, signal)

    input_norm = ImageUtils.normalize(signal, 1.0, np.float32)  ## for saving
    input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,signal,gv.patch_size,xy_step,z_step)
    roi_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,roi.roi,gv.patch_size,xy_step,z_step)
    mask_patchs = []
    for i in range(input_patchs.shape[0]):
        mask_patchs.append(interperter(input_patchs[i],roi_patchs[i]))
    mask_patchs = np.array(mask_patchs)
    weights = _get_weights(input_patchs[0].shape)
    mask , d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[mask_patchs,np.ones_like(input_patchs)],weights,signal.shape,gv.patch_size,xy_step,z_step)
    
    del input_patchs
    del mask_patchs
    del weights
    
    mask_norm = np.nan_to_num(mask/d)
    mask_norm = ImageUtils.normalize(mask_norm + 0.0001, 1.0, np.float32)
    alpha = 0.2
    full_cam = np.zeros(shape=(*input_norm.shape[:-1], 3))
    for j in range(input_norm.shape[0]):
        img = input_norm[j, :, :]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_norm[j, :, :]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = np.float32(np.dstack([img, img, img]))
        cv2.addWeighted(heatmap, alpha, cam, 1 - alpha, 0, cam)
        full_cam[j, :, :] = cam
    mask = np.uint8(255 * full_cam)
    return mask, mask_norm

def save_mask(root_dir,mask, layer_name, roi_mode, roi_args, method):
    mask = np.float16(mask)
    roi = None
    if roi_mode == "full":
        roi = roi_mode
    else:
        roi = "{}_{}".format(roi_mode,str(roi_args))
    if method == "gradcam" or method=="X_gradcam":
        ImageUtils.imsave(
            mask, "{}/{}_layer_{}_{}.tiff".format(root_dir, method, layer_name, roi))
    else:
        ImageUtils.imsave(
            mask, "{}/{}_{}.tiff".format(root_dir, method, roi))

def predict(model, signal):
    tf.keras.backend.clear_session()
    _ = gc.collect() 
    px_start = 0
    py_start = 0
    pz_start = 0
    px_end = signal.shape[1]
    py_end = signal.shape[2]
    pz_end = signal.shape[0]
    xy_step = 64
    z_step = 16
    input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,signal,gv.patch_size,xy_step,z_step)
    predict_patchs = model.predict(input_patchs,batch_size=16)
    weights = _get_weights(input_patchs[0].shape)
    prediction , d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[predict_patchs,np.ones_like(predict_patchs)],weights,signal.shape,gv.patch_size,xy_step,z_step)
    
    del input_patchs
    del predict_patchs
    del weights
    
    return prediction/d

def collect_layers(model):

    layers = {}
    for layer in model.layers:
        if "conv" in layer.name:
            layers[layer.name] = layer
    return layers

def get_pearson_per_slice(pred,target,weights=None):
    scores = []
   
    for i in range(pred.shape[0]):
        if weights is not None:
            scores.append(pearson_corr(pred[i:i+1,:],target[i:i+1,:],weights[i:i+1,:]))
        else:
            scores.append(pearson_corr(pred[i:i+1,:],target[i:i+1,:]))
                      
    return scores

def plot_pearson_per_slice(scores, organelle):
    ax = plt.subplot(1, 1, 1)
    x = np.arange(len(scores))
    y1 = scores
    ax.plot(x,y1)
    ax.set_ylabel('Slice Pearson score')
    ax.set_xlabel('Slice')
    ax.set_title("{} Pearson score per slice".format(organelle))
    plt.figtext(0.9, 0.5, "Pearson score of image:{}".format(np.mean(scores)*len(scores)), fontsize=8)
    plt.legend(["Pearson score of slice"])
    plt.show()

def plot_pearson_graph(scores, organelle):
    ax = plt.subplot(1, 1, 1)
    x = np.arange(len(scores))
    y = scores
    m = np.ones(len(scores))*np.mean(scores)
    sp = m + np.ones(len(scores))*np.std(scores)
    sm  = m - np.ones(len(scores))*np.std(scores)
    ax.scatter(x,y)
    ax.plot(x,m,color='blue', linestyle='dashed',label="mean")
    ax.plot(x,sp,color='orange', linestyle='dashed',label="std")
    ax.plot(x,sm,color='orange', linestyle='dashed')
    ax.set_ylabel('Pearson score')
    ax.set_xlabel('Image number')
    ax.set_title("{} Pearson scores".format(organelle))
    plt.figtext(0.9, 0.5, "mean:{}\nstd:{}".format(np.mean(scores),np.std(scores)), fontsize=8)
    plt.legend()
    plt.show()