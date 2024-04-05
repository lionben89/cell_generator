from copy import deepcopy
import cv2
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
from metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
import global_vars as gv
import os

gv.target = "channel_dna"

gv.organelle = "Nucleolus-(Granular-Component)"

gv.train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(gv.organelle.replace(' ','-'))
gv.test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_test.csv".format(gv.organelle.replace(' ','-'))

norm_type = "std"
gv.patch_size = (32,128,128,1)

compound = None #"Staurosporine" #"Staurosporine" #"Paclitaxol" 
is_vehicle = True #"staurosporine" #None #"rapamycin" #"paclitaxol" #"blebbistatin" #None #"staurosporine"
if compound is not None:
    if is_vehicle:
        ds_path = "/sise/home/lionb/single_cell_training_from_segmentation_pertrub/{}_{}/image_list_test_{}.csv".format(gv.organelle,compound,"Vehicle")
    else:
        ds_path = "/sise/home/lionb/single_cell_training_from_segmentation_pertrub/{}_{}/image_list_test_{}.csv".format(gv.organelle,compound,compound)
else:
    ds_path = gv.test_ds_path
    
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

print("Model: ",gv.unet_model_path)
print("Organelle: ",gv.organelle)
print("Compound: ",compound)
print("Vehicle: ", is_vehicle)

test_dataset = DataGen(ds_path, gv.input, gv.target, batch_size=1, num_batches=1, patch_size=gv.patch_size, min_precentage=0, max_precentage=1, augment=False, norm_type=norm_type)

images = [0]
models = ["./unet_model_22_05_22_tj_128","./unet_model_22_05_22_ngc_128","./unet_model_22_05_22_ne_128","./unet_model_22_05_22_mito_128",\
        "./unet_model_22_05_22_microtubules_128","./unet_model_22_05_22_membrane_128","./unet_model_22_05_22_golgi_128",\
        "./unet_model_22_05_22_er_128","./unet_model_22_05_22_dna_128","./unet_model_22_05_22_bundles_128","./unet_model_22_05_22_actin_128"\
        ]
for model in models:
    gv.unet_model_path = model
    unet = keras.models.load_model(gv.unet_model_path)
    if (not os.path.exists("./figure1/{}/predictions".format(gv.unet_model_path[1:]))):
        os.makedirs("{}/predictions".format(gv.unet_model_path))
    pcc = 0
    for image_index in images:        
        # image_index = 1
        if (not os.path.exists("{}/predictions/{}".format(gv.unet_model_path,image_index))):
            os.makedirs("{}/predictions/{}".format(gv.unet_model_path,image_index))
        image_path = test_dataset.df.get_item(image_index,'path_tiff')
        input_image, input_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.input_col,0)
        target_image, target_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.target_col,0)
        # nuc_seg, nuc_seg_new_file_path = test_dataset.get_image_from_ssd(image_path,"dna_seg",0)
        pred_image, prediction_new_file_path = test_dataset.get_image_from_ssd(image_path, "prediction", 0)
        
        if (input_image is None or target_image is None or pred_image is None):

            image_ndarray = None
            image_ndarray = ImageUtils.image_to_ndarray(ImageUtils.imread(image_path))
            channel_index = int(test_dataset.df.get_item(image_index,test_dataset.input_col))
            input_image = ImageUtils.get_channel(image_ndarray,channel_index)
            channel_index = int(test_dataset.df.get_item(image_index,test_dataset.target_col))
            target_image = ImageUtils.get_channel(image_ndarray,channel_index)
            channel_index = int(test_dataset.df.get_item(image_index, "channel_dna"))
            pred_image = ImageUtils.get_channel(image_ndarray, channel_index)
            # channel_index = int(test_dataset.df.get_item(image_index, "dna_seg"))
            # nuc_seg = ImageUtils.get_channel(image_ndarray, channel_index)
            
            input_image = np.expand_dims(input_image[0], axis=-1)
            target_image = np.expand_dims(target_image[0], axis=-1)
            pred_image = np.expand_dims(pred_image[0], axis=-1)
            # nuc_seg = np.expand_dims(nuc_seg[0], axis=-1)
            
            if norm_type == "minmax":
                target_image = normalize(target_image,max_value=1.0,dtype=np.float32)
                input_image = normalize(input_image,max_value=1.0,dtype=np.float32)
                pred_image = normalize(pred_image,max_value=1.0,dtype=np.float32)
            else:
                target_mean = np.mean(target_image,dtype=np.float64)
                target_std = np.std(target_image,dtype=np.float64)
                target_image = (target_image-target_mean)/target_std
                
                max_var = np.max(input_image!=np.inf)
                input_image = np.where(input_image==np.inf,max_var,input_image)
                input_mean = np.mean(input_image,dtype=np.float64)
                input_std = np.std(input_image,dtype=np.float64)
                input_image = (input_image-input_mean)/input_std
                
                pred_mean = np.mean(pred_image,dtype=np.float64)
                pred_std = np.std(pred_image,dtype=np.float64)
                pred_image = (pred_image-pred_mean)/pred_std
        
        i=0
        j=0
        k=0
        prediction = np.zeros_like(target_image)
        d = np.zeros_like(target_image)+1e-4
        o = 32 #overlap in xy dim
        od = 16 #overlap in z dim
        # input_image = np.pad(input_image,((0,0),()))
        while i<=input_image.shape[0]-gv.patch_size[0]:
            while j<=input_image.shape[1]-gv.patch_size[1]:
                while k<=input_image.shape[2]-gv.patch_size[2]:
                    s = [(i,i+gv.patch_size[0]),(j,j+gv.patch_size[1]),(k,k+gv.patch_size[2])]
                    patch = slice_image(input_image,s)
                    pred_patch = slice_image(pred_image,s)
                    patch = ImageUtils.to_shape(patch,gv.patch_size,min_shape=gv.patch_size)
                    if predictors:
                        pred_patch = ImageUtils.to_shape(pred_patch,gv.patch_size,min_shape=gv.patch_size)
                        
                        # patch_p = unet(np.expand_dims([patch],axis=0))
                        patch_p = unet.unet(np.expand_dims(np.concatenate([patch,pred_patch],axis=-1),axis=0))
                    else:
                        patch_p = unet(np.expand_dims(patch,axis=0))
                    weights = _get_weights(patch_p.shape)
                    prediction[i:i+gv.patch_size[0],j:j+gv.patch_size[1],k:k+gv.patch_size[2]] += patch_p*weights #((std*patch_p)+mean)/1000
                    d[i:i+gv.patch_size[0],j:j+gv.patch_size[1],k:k+gv.patch_size[2]] += weights[0]
                    k+=o
                k=0
                j+=o
            j=0
            i+=od
        prediction_cut = (prediction/(d))[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)]
        target_cut = (target_image)[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)]
        input_cut = (input_image)[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)]
        # nuc_seg_cut = (nuc_seg)[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)]
        
        ImageUtils.imsave(input_cut.astype(np.float16),"{}/predictions/{}/input_patch_{}.tiff".format(gv.unet_model_path,image_index,image_index))
        ImageUtils.imsave(target_cut.astype(np.float16),"{}/predictions/{}/target_patch_{}.tiff".format(gv.unet_model_path,image_index,image_index))
        ImageUtils.imsave(prediction_cut.astype(np.float16),"{}/predictions/{}/prediction_patch_{}.tiff".format(gv.unet_model_path,image_index,image_index))
        # ImageUtils.imsave(nuc_seg_cut.astype(np.float16),"{}/predictions/{}/nuc_seg_patch_{}.tiff".format(gv.unet_model_path,image_index,image_index))
        p=pearson_corr(target_cut, prediction_cut)
        pcc+=p
        print("pearson corr for image:{} is :{}".format(image_index,p))

        print("prediction - mean:{}, std:{}".format(np.mean(prediction_cut,dtype=np.float64),np.std(prediction_cut,dtype=np.float64)))
        print("target - mean:{}, std:{}".format(np.mean(target_image,dtype=np.float64),np.std(target_image,dtype=np.float64)))        
    print("average pcc:{}".format(pcc/len(images)))