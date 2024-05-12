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

#Model to predict and compare to GT
gv.model_type = "UNET"
for_clf = (gv.model_type == "CLF")

predictors=None #True w_dna
gv.model_path = "./unet_model_22_05_22_dna_128b"

#Input and target channels in the image
gv.input = "channel_signal"
gv.target = "channel_dna"

#Organelle to predict the model upon
gv.organelle = "Nucleolus-(Granular-Component)" #"Tight-junctions" #Actin-filaments" #"Golgi" #"Microtubules" #"Endoplasmic-reticulum" 
#"Plasma-membrane" #"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)"

#Assemble the proper tarining csvs by the organelle, model type, and if the data is pertrubed or not
if gv.model_type == "CLF":
    gv.input = "channel_target"
    gv.target = "channel_target"
    gv.train_ds_path = "/home/lionb/cell_generator/image_list_train.csv"
    gv.test_ds_path = "/home/lionb/cell_generator/image_list_test.csv"
else:
    gv.train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(gv.organelle.replace(' ','-'))
    gv.test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_test.csv".format(gv.organelle.replace(' ','-'))
#if compound is not None then it will take pertrubed dataset
compound = None #"s-Nitro-Blebbistatin" #"s-Nitro-Blebbistatin" #"Staurosporine" #None #"s-Nitro-Blebbistatin" #None #"paclitaxol_vehicle" #None #"paclitaxol_vehicle" #"rapamycin" #"paclitaxol" #"blebbistatin" #""
#drug could be either the compound or Vehicle which is like DMSO (the unpertrubed data in the pertrubed dataset)
drug = compound #"Vehicle"
if compound is not None:
    ds_path = "/sise/home/lionb/single_cell_training_from_segmentation_pertrub/{}_{}/image_list_test_{}.csv".format(gv.organelle,compound,drug)
else:
    ds_path = gv.test_ds_path

gv.batch_size = 8
norm_type = "std" #"minmax"#"std"#
gv.patch_size = (32,128,128,1)

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

print("Model: ",gv.unet_model_path)
print("Organelle: ",gv.organelle)
print("Compound: ",compound)
print("Vehicle: ", drug)

#Method to create triangular scheme of weights to deal with overlap
# in patchs, prediction towards the center of the patch will have higher weight in the end prediction.
def _get_weights(shape):
    shape_in = shape
    shape = shape[1:]
    weights = 1
    for idx_d in range(len(shape)):
        slicey = [np.newaxis] * len(shape)
        slicey[idx_d] = slice(None)
        size = shape[idx_d]
        values = scipy.signal.triang(size)
        weights = weights * values[tuple(slicey)] #scipy.ndimage.gaussian_filter(values, sigma=1)[tuple(slicey)]
        
        #weights = weights * np.ones(size)[tuple(slicey)]
    return np.broadcast_to(weights, shape_in).astype(np.float32)

def normalize(image_ndarray,max_value=255,dtype=np.uint8) -> np.ndarray:
    image_ndarray = image_ndarray.astype(np.float64)
    max_var = np.max(image_ndarray!=np.inf)
    image_ndarray = np.where(image_ndarray==np.inf,max_var,image_ndarray)
    temp_image = image_ndarray-np.min(image_ndarray)
    return ((temp_image)/((np.max(temp_image))*max_value)).astype(dtype)

#Create a test dataset with 1 sample just to take the list of the images
test_dataset = DataGen(ds_path, gv.input, gv.target, batch_size=1, num_batches=1, patch_size=gv.patch_size, min_precentage=0, max_precentage=1, augment=False, norm_type=norm_type,predictors=predictors)


def slice_image(image_ndarray: np.ndarray, indexes: list) -> np.ndarray:
    n_dim = len(image_ndarray.shape)
    slices = [slice(None)] * n_dim
    for i in range(len(indexes)):
        slices[i] = slice(indexes[i][0], indexes[i][1])
    slices = tuple(slices)
    sliced_image = image_ndarray[slices]
    return sliced_image

if (gv.model_type == "VAE"):
    from models.VAE import *
    vae = keras.models.load_model(gv.model_path)

    if (not os.path.exists("{}/predictions".format(gv.model_path))):
        os.makedirs("{}/predictions".format(gv.model_path))
    n = test_dataset.__len__()
    ppc = 0
    out = 0
    for j in range(n):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]
        prediction = vae(patchs).numpy()
        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] + i
            if k < 10:
                ImageUtils.imsave(
                    patchs[i], "{}/predictions/input_patch_{}.tiff".format(gv.model_path, k))
                ImageUtils.imsave(
                    target_patchs[i], "{}/predictions/target_patch_{}.tiff".format(gv.model_path, k))
                ImageUtils.imsave(
                    prediction[i], "{}/predictions/prediction_patch_{}.tiff".format(gv.model_path, k))
            p = pearson_corr(target_patchs[i], prediction[i])

            ppc += p
            print("pearson correlation for image {}: {}".format(k, p))
    print("avg pearson correlation: {}".format(ppc/((patchs.shape[0]*n)-out)))

elif (gv.model_type == "AAE"):
    from models.AAE import *
    aae = keras.models.load_model(gv.model_path)

    if (not os.path.exists("{}/predictions".format(gv.model_path))):
        os.makedirs("{}/predictions".format(gv.model_path))
    n = test_dataset.__len__()
    ppc = 0
    out = 0
    for j in range(n):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]
        z = aae.encoder(patchs).numpy()
        prediction = aae.decoder(z).numpy()
        z[:, 56] = -2
        alteredm2 = aae.decoder(z).numpy()
        z[:, 56] = 2
        alteredp2 = aae.decoder(z).numpy()
        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] + i
            if k < 10:
                ImageUtils.imsave(
                    alteredm2[i]*255, "{}/predictions/altered_minus2_patch_{}.tiff".format(gv.model_path, k))
                ImageUtils.imsave(
                    alteredp2[i]*255, "{}/predictions/altered_plus2_patch_{}.tiff".format(gv.model_path, k))
                ImageUtils.imsave(
                    target_patchs[i]*255, "{}/predictions/target_patch_{}.tiff".format(gv.model_path, k))
                ImageUtils.imsave(
                    prediction[i]*255, "{}/predictions/prediction_patch_{}.tiff".format(gv.model_path, k))
            p = PSNR(target_patchs[i]*255, prediction[i]*255)

            ppc += p
            print("psnr correlation for image {}: {}".format(k, p))
    print("avg psnr correlation: {}".format(ppc/((patchs.shape[0]*n)-out)))

elif (gv.model_type == "AE"):
    from models.AE import *
    ae = keras.models.load_model(gv.model_path)

    if (not os.path.exists("{}/predictions".format(gv.model_path))):
        os.makedirs("{}/predictions".format(gv.model_path))
    n = test_dataset.__len__()
    ppc = 0
    out = 0
    for j in range(n):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]
        prediction = ae(patchs).numpy()
        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] + i
            if k < 10:
                ImageUtils.imsave(
                    patchs[i], "{}/predictions/input_patch_{}.tiff".format(gv.model_path, k))
                ImageUtils.imsave(
                    target_patchs[i], "{}/predictions/target_patch_{}.tiff".format(gv.model_path, k))
                ImageUtils.imsave(
                    prediction[i], "{}/predictions/prediction_patch_{}.tiff".format(gv.model_path, k))
            p = pearson_corr(target_patchs[i], prediction[i])

            ppc += p
            print("pearson correlation for image {}: {}".format(k, p))
    print("avg pearson correlation: {}".format(ppc/((patchs.shape[0]*n)-out)))

elif (gv.model_type == "CLF"):
    clf = keras.models.load_model(gv.clf_model_path)
    dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 4, num_batches = 8, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False, for_clf=for_clf)
    for i in range(20):
        clf.evaluate(dataset)
        
elif (gv.model_type == "UNET"):
    # images = [1]
    images = range(min(10,test_dataset.df.get_shape()[0]))
    unet = keras.models.load_model(gv.unet_model_path)
    if (not os.path.exists("{}/predictions".format(gv.unet_model_path))):
        os.makedirs("{}/predictions".format(gv.unet_model_path))
    pcc = 0
    for image_index in images:        
        # image_index = 1
        if (not os.path.exists("{}/predictions/{}".format(gv.unet_model_path,image_index))):
            os.makedirs("{}/predictions/{}".format(gv.unet_model_path,image_index))
        image_path = test_dataset.df.get_item(image_index,'path_tiff')
        input_image, input_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.input_col)
        target_image, target_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.target_col)
        # nuc_seg, nuc_seg_new_file_path = test_dataset.get_image_from_ssd(image_path,"dna_seg")
        pred_image, prediction_new_file_path = test_dataset.get_image_from_ssd(image_path, "prediction")
        
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

elif (gv.model_type == "MG"):
    import tensorflow_addons as tfa
    from models.MaskInterpreter import *
    mg = keras.models.load_model(gv.mg_model_path)
    dir_path = "predictions"
    if (not os.path.exists("{}/{}".format(gv.mg_model_path,dir_path))):
        os.makedirs("{}/{}".format(gv.mg_model_path,dir_path))
    
    for image_index in [0,1,2]:#range(int(test_dataset.df.get_shape()[0]*0.00),int(test_dataset.df.get_shape()[0]*0.01)): # range(1):#range(test_dataset.df.get_shape()[0]):
        # image_index = 1
        if (not os.path.exists("{}/{}/{}".format(gv.mg_model_path,dir_path,image_index))):
            os.makedirs("{}/{}/{}".format(gv.mg_model_path,dir_path,image_index))
        image_path = test_dataset.df.get_item(image_index,'path_tiff')
        input_image, input_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.input_col)
        target_image, target_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.target_col)
        seg_image = None
        nuc_image = None
        ths = [-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #[0.0]#[1.00,0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        if (input_image is None or target_image is None or seg_image is None):

            organelle_mask_plus = ImageUtils.image_to_ndarray(ImageUtils.imread("/sise/home/lionb/mg_model_mito_29_05_22pcc1_0_of/predictions_cell/0/mask_binary_organelle_plus.tif"))
            organelle_mask_plus = np.expand_dims(organelle_mask_plus[0], axis=-1)
            image_ndarray = None
            image_ndarray = ImageUtils.image_to_ndarray(ImageUtils.imread(image_path))
            channel_index = int(test_dataset.df.get_item(image_index,test_dataset.input_col))
            input_image = ImageUtils.get_channel(image_ndarray,channel_index)
            channel_index = int(test_dataset.df.get_item(image_index,test_dataset.target_col))
            target_image = ImageUtils.get_channel(image_ndarray,channel_index)
            input_image = np.expand_dims(input_image[0], axis=-1)
            target_image = np.expand_dims(target_image[0], axis=-1)
            target_mean = np.mean(target_image,dtype=np.float64)
            target_std = np.std(target_image,dtype=np.float64)
            target_image = (target_image-target_mean)/target_std
            max_var = np.max(input_image!=np.inf)
            input_image = np.where(input_image==np.inf,max_var,input_image)
            input_mean = np.mean(input_image,dtype=np.float64)
            input_std = np.std(input_image,dtype=np.float64)
            input_image = (input_image-input_mean)/input_std
            channel_index = int(test_dataset.df.get_item(image_index,"structure_seg"))
            seg_image = ImageUtils.get_channel(image_ndarray,channel_index)
            seg_image = np.expand_dims(seg_image[0], axis=-1)
            channel_index = int(test_dataset.df.get_item(image_index,"channel_dna"))
            nuc_image = ImageUtils.get_channel(image_ndarray,channel_index)
            nuc_image = np.expand_dims(nuc_image[0], axis=-1)
            channel_index = int(test_dataset.df.get_item(image_index,"channel_membrane"))
            mem_image = ImageUtils.get_channel(image_ndarray,channel_index)
            mem_image = np.expand_dims(mem_image[0], axis=-1)
        for th in ths:
            if (not os.path.exists("{}/{}/{}/{}".format(gv.mg_model_path,dir_path,image_index,th))):
                os.makedirs("{}/{}/{}/{}".format(gv.mg_model_path,dir_path,image_index,th))
            o = 32
            od = 16
            weights = None
            center_xy = [243,192]
            i=0
            j=center_xy[0]-128-o
            k=center_xy[1]-128-o                        
            s_full = [(0,input_image.shape[0]),(center_xy[0]-128-o,center_xy[0]+128+o),(center_xy[1]-128-o,center_xy[1]+128+o)]
            sliced_input = slice_image(input_image,s_full)
            sliced_target = slice_image(target_image,s_full)
            sliced_nuc = slice_image(nuc_image,s_full)
            sliced_mem = slice_image(mem_image,s_full)
            prediction = np.zeros_like(sliced_input)
            mask = np.zeros_like(sliced_input)
            adapted_input = np.zeros_like(sliced_input)
            adapted_prediction = np.zeros_like(sliced_input)
            d = np.zeros_like(sliced_input)+1e-4
            d2 = np.zeros_like(sliced_input)+1e-4
             
            # while i<=input_image.shape[0]-gv.patch_size[0]:
            #     while j<=input_image.shape[1]-gv.patch_size[1]:
            #         while k<=input_image.shape[2]-gv.patch_size[2]:
            while i<=input_image.shape[0]-gv.patch_size[0]:
                while j<=center_xy[0]+128+o-gv.patch_size[1]:
                    while k<=center_xy[1]+128+o-gv.patch_size[2]:            
                        s = [(i,i+gv.patch_size[0]),(j,j+gv.patch_size[1]),(k,k+gv.patch_size[2])]
                        patch = slice_image(input_image,s)
                        patch = ImageUtils.to_shape(patch,gv.patch_size,min_shape=gv.patch_size)
                        seg_patch = slice_image(seg_image,s)
                        patch_p = mg.unet(np.expand_dims(patch,axis=0))
                        mask_p = mg(np.expand_dims(patch,axis=0))
                        # mask_p2 = tf.where(mask_p>th,mask_p,0.0) ## th
                        mask_p2 = tf.where(tf.math.logical_and(mask_p>th,mask_p<=(th+0.1)),0.0,mask_p) ## th
                        # mask_p = np.expand_dims(seg_patch/255.0,axis=0) #tf.where(seg_patch>0,mask_p,tf.constant(0.0,dtype=tf.float64))
                        # mask_p = tf.nn.max_pool3d(mask_p, ksize=3, strides=1, padding="SAME", name='dilation3D')
                        # mask_p = tf.cast(mask_p,dtype=tf.float64)
                        normal_noise = tf.random.normal(tf.shape(mask_p),stddev=1.0,dtype=tf.float64)*1.0
                        mask_noise = (normal_noise*(1-mask_p2))
                        adapted_input_p = patch+mask_noise
                        adapted_prediction_p = mg.unet(adapted_input_p)
                        if weights is None:
                            weights = _get_weights(patch_p.shape)
                        j_x = j-(center_xy[0]-128-o)
                        k_x = k-(center_xy[1]-128-o)
                        prediction[i:i+gv.patch_size[0],j_x:j_x+gv.patch_size[1],k_x:k_x+gv.patch_size[2]] += patch_p*weights 
                        mask[i:i+gv.patch_size[0],j_x:j_x+gv.patch_size[1],k_x:k_x+gv.patch_size[2]] += mask_p*weights 
                        adapted_input[i:i+gv.patch_size[0],j_x:j_x+gv.patch_size[1],k_x:k_x+gv.patch_size[2]] += adapted_input_p*weights 
                        adapted_prediction[i:i+gv.patch_size[0],j_x:j_x+gv.patch_size[1],k_x:k_x+gv.patch_size[2]] += adapted_prediction_p*weights 
                        d[i:i+gv.patch_size[0],j_x:j_x+gv.patch_size[1],k_x:k_x+gv.patch_size[2]] += weights[0]
                        d2[i:i+gv.patch_size[0],j_x:j_x+gv.patch_size[1],k_x:k_x+gv.patch_size[2]] += 1
                        k+=o
                    k=center_xy[1]-128-o  
                    j+=o
                j=center_xy[0]-128-o
                i+=od
                
            mask2 = mask/d    
            mask2 = tf.where(tf.math.logical_and((mask2)>th,(mask2)<=(th+0.1)),0.0,mask2).numpy()
            # mask2 = tf.where(mask>th,mask,0.0).numpy()
            ImageUtils.imsave(sliced_input,"{}/{}/{}/{}/input_patch_{}.tiff".format(gv.mg_model_path,dir_path,image_index,th,image_index))
            ImageUtils.imsave(sliced_target,"{}/{}/{}/{}/target_patch_{}.tiff".format(gv.mg_model_path,dir_path,image_index,th,image_index))
            ImageUtils.imsave(sliced_nuc,"{}/{}/{}/{}/nuc_patch_{}.tiff".format(gv.mg_model_path,dir_path,image_index,th,image_index))
            ImageUtils.imsave(sliced_mem,"{}/{}/{}/{}/mem_patch_{}.tiff".format(gv.mg_model_path,dir_path,image_index,th,image_index))
            ImageUtils.imsave((prediction/(d))[:,:,:],"{}/{}/{}/{}/unet_prediction_{}.tiff".format(gv.mg_model_path,dir_path,image_index,th,image_index))
            ImageUtils.imsave(mask2[:,:,:],"{}/{}/{}/{}/mask_{}.tiff".format(gv.mg_model_path,dir_path,image_index,th,image_index))
            ImageUtils.imsave((adapted_prediction/(d))[:,:,:],"{}/{}/{}/{}/adapted_unet_prediction_{}.tiff".format(gv.mg_model_path,dir_path,image_index,th,image_index))
            ImageUtils.imsave((adapted_input/(d))[:,:,:],"{}/{}/{}/{}/adapted_input_{}.tiff".format(gv.mg_model_path,dir_path,image_index,th,image_index))
            print(th)
            print("pearson corr for image:{} is :{}, mask ratio:{}".format(image_index,pearson_corr((prediction/d)[:,:,:], (adapted_prediction/(d))[:,:,:]),np.mean(mask2,dtype=np.float64)))

elif (gv.model_type == "RC"):
    rc = keras.models.load_model(gv.rc_model_path)
    if (not os.path.exists("{}/predictions".format(gv.rc_model_path))):
        os.makedirs("{}/predictions".format(gv.rc_model_path))
    n = test_dataset.__len__()
    ppc = 0
    pred_ppc = 0
    for j in range(n):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]
        x = rc(patchs)
        prediction_score = x[0].numpy()
        features = x[1].numpy()
        prediction = rc.unet(patchs).numpy()

        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] + i
            if k < 10:
                ImageUtils.imsave(
                    patchs[i], "{}/predictions/input_patch_{}.tiff".format(gv.rc_model_path, k))
                ImageUtils.imsave(
                    target_patchs[i], "{}/predictions/target_patch_{}.tiff".format(gv.rc_model_path, k))
                ImageUtils.imsave(
                    prediction[i], "{}/predictions/prediction_{}.tiff".format(gv.rc_model_path, k))
            p = pearson_corr(target_patchs[i], prediction[i])
            pred_ppc += prediction_score[i][0]
            ppc += p

            print("pearson correlation for image {}: {}, predicted:{}".format(
                k, p, prediction_score[i][0]))
    print("avg pearson correlation: {}, predicted:{}".format(
        ppc/(n*patchs.shape[0]), pred_ppc/(n*patchs.shape[0])))

elif (gv.model_type == "PM"):
    pm = keras.models.load_model(gv.pm_model_path)
    if (not os.path.exists("{}/predictions".format(gv.pm_model_path))):
        os.makedirs("{}/predictions".format(gv.pm_model_path))
    unet = keras.models.load_model("./unet_model_x_mse_2_3_nobn_ne")
    len = test_dataset.__len__()
    total_ba = []
    for j in range(len):
        patchs = test_dataset.__getitem__(j)[0]
        labels = test_dataset.__getitem__(j)[1]
        # unet_input = unpatchify(patchs[0].reshape(1,4,4,16,64,64),(16,256,256)).reshape(gv.patch_size)
        # unet_image =unet(np.expand_dims(unet_input,axis=0)).numpy()[0]
        # unet_patchs = patchify(unet_image,(16,64,64,1),step=64).reshape(16,16,64,64,1)
        predictions = pm([patchs[0],patchs[1]]).numpy()
        # unet_predictions = pm([patchs[0],unet_patchs]).numpy()
        total_ba.append(tf.reduce_mean(keras.metrics.binary_accuracy(labels,predictions)))
        for i in range(patchs[0].shape[0]):
            k = j*patchs[0].shape[0] + i
            # prediction = pm([patchs[0][0:1],patchs[1][i:i+1]]).numpy()
            if k < 32:
                # ImageUtils.imsave(
                    # unet_patchs[i], "{}/predictions/unet_prediction_patch_{}.tiff".format(gv.pm_model_path, k))
                ImageUtils.imsave(
                    patchs[0][i], "{}/predictions/input_patch_{}.tiff".format(gv.pm_model_path, k))
                ImageUtils.imsave(
                    patchs[1][i], "{}/predictions/target_patch_{}.tiff".format(gv.pm_model_path, k))
            
            # pear = pearson_corr(target_patchs[i], prediction_patchs[i])
            # s_pear = pearson_corr(target_patchs[i], s_prediction_patchs[i])
            print("score for pair {}: prediction:{}, label:{}".format(
                k, predictions[i], labels[i]))
    print("total accuracy is :",np.average(total_ba))
    

