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
from skimage.filters import threshold_li
# from patchify import unpatchify,patchify
# from sklearn.neighbors import KernelDensity

# tf.compat.v1.enable_eager_execution()

gv.model_type = "UNET"
for_clf = (gv.model_type == "CLF")
predictors=None #True w_dna
gv.unet_model_path = "./unet_model_22_05_22_actin_128" #_48_64_64"#"unet_model_22_05_22_actin_128" #unet_model_22_05_22_membrane_w_dna "./unet_model_22_05_22_membrane_128" #"./unet_model_22_05_22_actin_128p_save_bs4-1"
gv.clf_model_path = "./clf_model_14_12_22-1"
gv.organelle = "Actin-filaments" #"Tight-junctions" #Actin-filaments" #"Golgi" #"Microtubules" #"Endoplasmic-reticulum" 
#"Plasma-membrane" #"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)"
if gv.model_type == "CLF":
    gv.input = "channel_target"
    gv.target = "channel_target"
    gv.train_ds_path = "/home/lionb/cell_generator/image_list_train.csv"
    gv.test_ds_path = "/home/lionb/cell_generator/image_list_test.csv"
else:
    gv.train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(gv.organelle)
    gv.test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_test.csv".format(gv.organelle)
norm_type = "std" #"minmax"#"std"#
# gv.patch_size = (48,64,64,1)
gv.patch_size = (32,128,128,1)

compound = None #"staurosporine" #None #"rapamycin" #"paclitaxol" #"blebbistatin" #None #"staurosporine"
if compound is not None:
    ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}_{}/image_list_test.csv".format(gv.organelle,compound)
else:
    ds_path = gv.train_ds_path
    
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

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

test_dataset = DataGen(ds_path, gv.input, gv.target, batch_size=1, num_batches=1, patch_size=gv.patch_size, min_precentage=0, max_precentage=1, augment=False, norm_type=norm_type,predictors=predictors)
# test_dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=0.9, augment=False)

def euclidean_distance(x,y):
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
    
def heatmap(img, mask):
    alpha = 0.5
    mask = mask[7] - np.min(mask[7])
    mask = mask / np.max(mask)
    img = img[7]
    heatmap = cv2.applyColorMap(
        np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, 2]
    heatmap = np.float32(
        np.dstack([heatmap, np.zeros_like(mask), np.zeros_like(mask)]))/255
    # heatmap = np.float32(heatmap) / 255
    cam = np.float32(np.dstack([img, img, img]))
    cv2.addWeighted(heatmap, alpha, cam, 1 - alpha, 0, cam)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
    # return heatmap


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

elif (gv.model_type == "L2L"):
    from models.Latent2Latent import *
    l2l = keras.models.load_model(gv.latent_to_latent_model_path)

    patchs = test_dataset.__getitem__(0)[0]
    target_patchs = test_dataset.__getitem__(0)[1]
    z_true = l2l.target_encoder(target_patchs)
    z = l2l.input_encoder(patchs)
    prediction = l2l.target_decoder(z).numpy()

    for i in range(patchs.shape[0]):
        ImageUtils.imsave(patchs[i], "input_patch_{}.tiff".format(i))
        ImageUtils.imsave(target_patchs[i], "target_patch_{}.tiff".format(i))
        ImageUtils.imsave(
            prediction[i], "l2l_prediction_patch_{}.tiff".format(i))
        print("latent dim diff for image {}: {}".format(
            i, sum(np.abs(z_true[i]-z[i]))))
        print("pearson correlation for image {}: {}".format(
            i, pearson_corr(target_patchs[i], prediction[i])))

elif (gv.model_type == "L2LRes"):
    from models.Latent2LatentRes import *
    l2l = keras.models.load_model(gv.latent_to_latent_model_path)

    patchs = test_dataset.__getitem__(0)[0]
    target_patchs = test_dataset.__getitem__(0)[1]
    prediction = l2l(patchs)

    for i in range(patchs.shape[0]):
        ImageUtils.imsave(patchs[i], "input_patch_{}.tiff".format(i))
        ImageUtils.imsave(target_patchs[i], "target_patch_{}.tiff".format(i))
        ImageUtils.imsave(
            prediction[i], "l2lres_prediction_patch_{}.tiff".format(i))
        print("pearson correlation for image {}: {}".format(
            i, pearson_corr(target_patchs[i], prediction[i])))

elif (gv.model_type == "CLF"):
    clf = keras.models.load_model(gv.clf_model_path)
    dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 4, num_batches = 8, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False, for_clf=for_clf)
    for i in range(20):
        clf.evaluate(dataset)
        
elif (gv.model_type == "UNET"):
    images = [1]
    # images = range(test_dataset.df.get_shape()[0])
    unet = keras.models.load_model(gv.unet_model_path)
    if (not os.path.exists("{}/predictions".format(gv.unet_model_path))):
        os.makedirs("{}/predictions".format(gv.unet_model_path))
    pcc = 0
    for image_index in images:        
        # image_index = 1
        if (not os.path.exists("{}/predictions/{}".format(gv.unet_model_path,image_index))):
            os.makedirs("{}/predictions/{}".format(gv.unet_model_path,image_index))
        image_path = test_dataset.df.get_item(image_index,'path_tiff')
        input_image, input_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.input_col,0)
        target_image, target_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.target_col,0)
        nuc_seg, nuc_seg_new_file_path = test_dataset.get_image_from_ssd(image_path,"dna_seg",0)
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
            channel_index = int(test_dataset.df.get_item(image_index, "dna_seg"))
            nuc_seg = ImageUtils.get_channel(image_ndarray, channel_index)
            
            input_image = np.expand_dims(input_image[0], axis=-1)
            target_image = np.expand_dims(target_image[0], axis=-1)
            pred_image = np.expand_dims(pred_image[0], axis=-1)
            nuc_seg = np.expand_dims(nuc_seg[0], axis=-1)
            
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
        nuc_seg_cut = (nuc_seg)[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)]
        
        ImageUtils.imsave(input_cut.astype(np.float16),"{}/predictions/{}/input_patch_{}.tiff".format(gv.unet_model_path,image_index,image_index))
        ImageUtils.imsave(target_cut.astype(np.float16),"{}/predictions/{}/target_patch_{}.tiff".format(gv.unet_model_path,image_index,image_index))
        ImageUtils.imsave(prediction_cut.astype(np.float16),"{}/predictions/{}/prediction_patch_{}.tiff".format(gv.unet_model_path,image_index,image_index))
        ImageUtils.imsave(nuc_seg_cut.astype(np.float16),"{}/predictions/{}/nuc_seg_patch_{}.tiff".format(gv.unet_model_path,image_index,image_index))
        p=pearson_corr(target_cut, prediction_cut)
        pcc+=p
        print("pearson corr for image:{} is :{}".format(image_index,p))
  
        print("prediction - mean:{}, std:{}".format(np.mean(prediction_cut,dtype=np.float64),np.std(prediction_cut,dtype=np.float64)))
        print("target - mean:{}, std:{}".format(np.mean(target_image,dtype=np.float64),np.std(target_image,dtype=np.float64)))        
    print("average pcc:{}".format(pcc/len(images)))
    
elif (gv.model_type == "SG"):
    from models.SampleGenerator import *
    sg = keras.models.load_model(gv.sg_model_path)
    if (not os.path.exists("{}/predictions".format(gv.sg_model_path))):
        os.makedirs("{}/predictions".format(gv.sg_model_path))
    n = test_dataset.__len__()
    ppc = 0
    out = 0
    for j in range(n):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]
        target_unet = sg.unet(patchs).numpy()
        z = sg.aae.encoder(target_unet).numpy()
        z[:, 56] = -2
        target_aae = sg.aae.decoder(z).numpy()
        prediction = sg.generator([patchs, z]).numpy()
        prediction_unet = sg.unet(prediction).numpy()

        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] + i
            if k < 10:
                ImageUtils.imsave(
                    patchs[i], "{}/predictions/input_patch_{}.tiff".format(gv.sg_model_path, k))
                ImageUtils.imsave(
                    target_patchs[i], "{}/predictions/target_patch_{}.tiff".format(gv.sg_model_path, k))
                ImageUtils.imsave(
                    prediction[i], "{}/predictions/adapted_input_patch_{}.tiff".format(gv.sg_model_path, k))
                ImageUtils.imsave(
                    target_aae[i], "{}/predictions/adapted_output_patch_{}.tiff".format(gv.sg_model_path, k))
                ImageUtils.imsave(
                    prediction_unet[i], "{}/predictions/predicted_adapted_output_patch_{}.tiff".format(gv.sg_model_path, k))
                ImageUtils.imsave(
                    target_unet[i], "{}/predictions/target_unet_output_patch_{}.tiff".format(gv.sg_model_path, k))
            p = pearson_corr(target_aae[i], prediction_unet[i])

            ppc += p
            print("pearson correlation for image {}: {}".format(k, p))
    print("avg pearson correlation: {}".format(ppc/((patchs.shape[0]*n)-out)))

elif (gv.model_type == "ShG"):
    shg = keras.models.load_model(gv.shg_model_path)
    if (not os.path.exists("{}/predictions".format(gv.shg_model_path))):
        os.makedirs("{}/predictions".format(gv.shg_model_path))
        
    d_norm = None
    for image_index in range(test_dataset.df.get_shape()[0]):
        # image_index = 1
        print("predicting: ",image_index)
        image_path = test_dataset.df.get_item(image_index,'path_tiff')
        input_image, input_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.input_col,0)
        target_image, target_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.target_col,0)
        
        alt_image_index = image_index#int(np.random.random_integers(0,test_dataset.df.get_shape()[0]-1))
        alt_image_path = test_dataset.df.get_item(alt_image_index,'path_tiff')
        alt_input_image, alt_input_new_file_path = test_dataset.get_image_from_ssd(alt_image_path,test_dataset.input_col,0)
        alt_target_image, alt_target_new_file_path = test_dataset.get_image_from_ssd(alt_image_path,test_dataset.target_col,0)
        if (input_image is None or target_image is None):

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
            
        if (alt_input_image is None or alt_target_image is None):
            alt_image_ndarray = None
            alt_image_ndarray = ImageUtils.image_to_ndarray(ImageUtils.imread(alt_image_path))
            channel_index = int(test_dataset.df.get_item(alt_image_index,test_dataset.input_col))
            alt_input_image = ImageUtils.get_channel(alt_image_ndarray,channel_index)
            channel_index = int(test_dataset.df.get_item(alt_image_index,test_dataset.target_col))
            alt_target_image = ImageUtils.get_channel(alt_image_ndarray,channel_index)
            alt_input_image = np.expand_dims(alt_input_image[0], axis=-1)
            alt_target_image = np.expand_dims(alt_target_image[0], axis=-1)
            alt_target_mean = np.mean(alt_target_image,dtype=np.float64)
            alt_target_std = np.std(alt_target_image,dtype=np.float64)
            alt_target_image = (alt_target_image-alt_target_mean)/alt_target_std
            alt_max_var = np.max(alt_input_image!=np.inf)
            alt_input_image = np.where(alt_input_image==np.inf,alt_max_var,alt_input_image)
            alt_input_mean = np.mean(alt_input_image,dtype=np.float64)
            alt_input_std = np.std(alt_input_image,dtype=np.float64)
            alt_input_image = (alt_input_image-alt_input_mean)/alt_input_std            

        i=0
        j=0
        k=0
        prediction = np.zeros_like(target_image)
        adapted_prediction = np.zeros_like(target_image)
        adapted_unet_prediction = np.zeros_like(target_image)
        d = np.zeros_like(target_image)+1e-4
        o = 64
        od=16
        # input_image = np.pad(input_image,((0,0),()))
        while i<=input_image.shape[0]-gv.patch_size[0]:
            while j<=input_image.shape[1]-gv.patch_size[1]:
                while k<=input_image.shape[2]-gv.patch_size[2]:
                    s = [(i,i+gv.patch_size[0]),(j,j+gv.patch_size[1]),(k,k+gv.patch_size[2])]
                    patch = slice_image(input_image,s)
                    patch = ImageUtils.to_shape(patch,gv.patch_size,min_shape=gv.patch_size)
                    target_patch = slice_image(alt_target_image,s)
                    target_patch = ImageUtils.to_shape(target_patch,gv.patch_size,min_shape=gv.patch_size)
                    alt_patch = slice_image(alt_input_image,s)
                    alt_patch = ImageUtils.to_shape(alt_patch,gv.patch_size,min_shape=gv.patch_size)
                    patch_p_unet = shg.unet(np.expand_dims(alt_patch,axis=0))
                    sh_patch_p = shg.generator([np.expand_dims(patch,axis=0),patch_p_unet])
                    sh_patch_p_unet = shg.unet(sh_patch_p)
                    
                    weights = _get_weights(sh_patch_p.shape)
                    
                    adapted_prediction[i:i+gv.patch_size[0],j:j+gv.patch_size[1],k:k+gv.patch_size[2]] += sh_patch_p*weights ##adapted input
                    adapted_unet_prediction[i:i+gv.patch_size[0],j:j+gv.patch_size[1],k:k+gv.patch_size[2]] += sh_patch_p_unet*weights ##adapted unet
                    prediction[i:i+gv.patch_size[0],j:j+gv.patch_size[1],k:k+gv.patch_size[2]] += patch_p_unet*weights ##original unet
                    
                    d[i:i+gv.patch_size[0],j:j+gv.patch_size[1],k:k+gv.patch_size[2]] += weights[0]
                    k+=o
                k=0
                j+=o
            j=0
            i+=od
        
        ImageUtils.imsave(input_image,"{}/predictions/input_patch_{}.tiff".format(gv.shg_model_path,image_index))
        # ImageUtils.imsave(target_image,"{}/predictions/target_patch_{}.tiff".format(gv.shg_model_path,image_index))
        ImageUtils.imsave(alt_target_image,"{}/predictions/alt_target_patch_{}.tiff".format(gv.shg_model_path,image_index))
        ImageUtils.imsave((prediction/(d))[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)],"{}/predictions/prediction_patch_{}.tiff".format(gv.shg_model_path,image_index))
        ImageUtils.imsave((adapted_prediction/(d))[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)],"{}/predictions/adapted_input_patch_{}.tiff".format(gv.shg_model_path,image_index))
        ImageUtils.imsave((adapted_unet_prediction/(d))[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)],"{}/predictions/adapted_prediction_patch_{}.tiff".format(gv.shg_model_path,image_index))
        try:
            print("pearson corr for adapted image:{} is :{}".format(image_index,pearson_corr(alt_target_image[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)], (adapted_unet_prediction/(d))[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)])))
            print("pearson corr for alt image:{} is :{}".format(image_index,pearson_corr(alt_target_image[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)], (prediction/(d))[:-1*(prediction.shape[0]%od),:-1*(prediction.shape[1]%o),:-1*(prediction.shape[2]%o)])))
        except:
            print("Error calculation pcc")
    
elif (gv.model_type == "ShGD"):
    shg = keras.models.load_model(gv.shg_model_path)
    if (not os.path.exists("{}/predictions".format(gv.shg_model_path))):
        os.makedirs("{}/predictions".format(gv.shg_model_path))
    n = test_dataset.__len__()
    ppc = 0
    out = 0
    for j in range(n):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]

        target_unet = shg.unet(patchs).numpy()

        prediction = shg.generator(patchs).numpy()

        prediction_unet = shg.unet(prediction).numpy()

        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] + i
            if k < 10:
                ImageUtils.imsave(
                    patchs[i], "{}/predictions/input_patch_{}.tiff".format(gv.shg_model_path, k))
                ImageUtils.imsave(
                    target_patchs[i], "{}/predictions/target_patch_{}.tiff".format(gv.shg_model_path, k))
                ImageUtils.imsave(
                    prediction[i], "{}/predictions/adapted_input_patch_{}.tiff".format(gv.shg_model_path, k))
                ImageUtils.imsave(
                    prediction_unet[i], "{}/predictions/predicted_adapted_output_patch_{}.tiff".format(gv.shg_model_path, k))
                ImageUtils.imsave(
                    target_unet[i], "{}/predictions/target_unet_output_patch_{}.tiff".format(gv.shg_model_path, k))
            p = pearson_corr(target_unet[i], prediction_unet[i])

            ppc += p
            print("pearson correlation for image {}: {}".format(k, p))

elif (gv.model_type == "EAM"):
    eam = keras.models.load_model(gv.eam_model_path)
    if (not os.path.exists("{}/predictions".format(gv.eam_model_path))):
        os.makedirs("{}/predictions".format(gv.eam_model_path))
    n = test_dataset.__len__()
    ppc = 0
    ppc1 = 0
    out = 0
    for j in range(n):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]
        target_unet = eam.unet(patchs).numpy()
        prediction = eam.generator(patchs).numpy()
        prediction_unet = eam.unet(prediction).numpy()

        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] + i
            if k < 10:
                ImageUtils.imsave(
                    patchs[i], "{}/predictions/input_patch_{}.tiff".format(gv.eam_model_path, k))
                ImageUtils.imsave(
                    target_patchs[i], "{}/predictions/target_patch_{}.tiff".format(gv.eam_model_path, k))
                ImageUtils.imsave(
                    prediction[i], "{}/predictions/adapted_input_patch_{}.tiff".format(gv.eam_model_path, k))
                ImageUtils.imsave(
                    prediction_unet[i], "{}/predictions/predicted_adapted_output_patch_{}.tiff".format(gv.eam_model_path, k))
                ImageUtils.imsave(
                    target_unet[i], "{}/predictions/target_unet_output_patch_{}.tiff".format(gv.eam_model_path, k))
                ImageUtils.imsave(np.square(
                    patchs[i]-prediction[i]), "{}/predictions/diff_patch_{}.tiff".format(gv.eam_model_path, k))
            p = dice(target_patchs[i], np.round(prediction_unet[i]))
            p1 = dice(target_patchs[i], np.round(target_unet[i]))

            ppc += p
            ppc1 += p1
            print(
                "dice correlation for image {}: {} and original score:{}".format(k, p, p1))
    print("avg dice correlation: {} and original score:{}".format(
        ppc/(n*patchs.shape[0]), ppc1/(n*patchs.shape[0])))

elif (gv.model_type == "ZG"):
    zg = keras.models.load_model(gv.zg_model_path)
    if (not os.path.exists("{}/predictions".format(gv.zg_model_path))):
        os.makedirs("{}/predictions".format(gv.zg_model_path))
    n = test_dataset.__len__()
    ppc = 0
    ppc1 = 0
    out = 0
    for j in range(n):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]
        target_unet = zg.unet(patchs).numpy()
        prediction = zg.generator(patchs).numpy()
        prediction_unet = zg.unet(prediction).numpy()

        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] + i
            if k < 10:
                ratio = np.zeros_like(patchs[i])
                input_ratio = np.zeros_like(patchs[i])

                for r in range(ratio.shape[0]):
                    avg = np.average(patchs[i, r])
                    ratio[r] = prediction[i, r]/(patchs[i, r]+0.0001)
                    ratio[r] = ratio[r] - np.min(ratio[r])
                    ratio[r] = ratio[r] / np.max(ratio[r])
                    th = threshold_li(ratio[r])
                    # th = np.percentile(prediction[i,r], 50)
                    input_ratio[r] = np.where(ratio[r] < th, patchs[i, r], avg)
                prediction_ratio = zg.unet(
                    np.expand_dims(input_ratio, axis=0)).numpy()
                ImageUtils.imsave(
                    patchs[i], "{}/predictions/input_patch_{}.tiff".format(gv.zg_model_path, k))
                ImageUtils.imsave(
                    target_patchs[i], "{}/predictions/target_patch_{}.tiff".format(gv.zg_model_path, k))
                ImageUtils.imsave(
                    prediction[i], "{}/predictions/adapted_input_patch_{}.tiff".format(gv.zg_model_path, k))
                ImageUtils.imsave(
                    prediction_unet[i], "{}/predictions/predicted_adapted_output_patch_{}.tiff".format(gv.zg_model_path, k))
                ImageUtils.imsave(
                    target_unet[i], "{}/predictions/target_unet_output_patch_{}.tiff".format(gv.zg_model_path, k))
                ImageUtils.imsave(
                    ratio, "{}/predictions/ratio_patch_{}.tiff".format(gv.zg_model_path, k))
                ImageUtils.imsave(
                    input_ratio, "{}/predictions/input_ratio_patch_{}.tiff".format(gv.zg_model_path, k))
                ImageUtils.imsave(
                    prediction_ratio, "{}/predictions/prediction_ratio_patch_{}.tiff".format(gv.zg_model_path, k))
                ImageUtils.imsave(heatmap(
                    patchs[i], prediction[i]), "{}/predictions/hm_{}.tiff".format(gv.zg_model_path, k))
            p = pearson_corr(target_unet[i], prediction_ratio[0])

            ppc += p
            print("pearson_corr for image {}: {}".format(k, p))
    print("avg pearson_corr: {}".format(ppc/(n*patchs.shape[0])))

elif (gv.model_type == "MG"):
    import tensorflow_addons as tfa
    from models.MaskGenerator import *
    mg = keras.models.load_model(gv.mg_model_path)
    dir_path = "predictions"
    if (not os.path.exists("{}/{}".format(gv.mg_model_path,dir_path))):
        os.makedirs("{}/{}".format(gv.mg_model_path,dir_path))
    
    for image_index in [0,1,2]:#range(int(test_dataset.df.get_shape()[0]*0.00),int(test_dataset.df.get_shape()[0]*0.01)): # range(1):#range(test_dataset.df.get_shape()[0]):
        # image_index = 1
        if (not os.path.exists("{}/{}/{}".format(gv.mg_model_path,dir_path,image_index))):
            os.makedirs("{}/{}/{}".format(gv.mg_model_path,dir_path,image_index))
        image_path = test_dataset.df.get_item(image_index,'path_tiff')
        input_image, input_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.input_col,0)
        target_image, target_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.target_col,0)
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
    
elif (gv.model_type == "SN"):
    from models.SN import *
    sn = keras.models.load_model(gv.sn_model_path,custom_objects={'ec': euclidean_distance})
    if (not os.path.exists("{}/predictions".format(gv.sn_model_path))):
        os.makedirs("{}/predictions".format(gv.sn_model_path))
    # unet = keras.models.load_model("./unet_model_x_mse_2_3_nobn_membrane")
    len = test_dataset.__len__()
    pi = 0
    ni = 0
    for j in range(len):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]
        # prediction_patchs = unet(patchs).numpy()
        # s_patchs = np.flip(patchs)
        # s_prediction_patchs = np.flip(prediction_patchs)
        s_target_patchs = np.flip(target_patchs)
        s_patchs = np.flip(patchs)
        p = sn.model([patchs, target_patchs]).numpy()
        n = sn.model([s_patchs, target_patchs]).numpy()

        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] + i
            if k < 32:
                # ImageUtils.imsave(
                #     s_prediction_patchs[i], "{}/predictions/s_prediction_patch_{}.tiff".format(gv.sn_model_path, k))
                ImageUtils.imsave(
                    patchs[i], "{}/predictions/input_patch_{}.tiff".format(gv.sn_model_path, k))
                ImageUtils.imsave(
                    target_patchs[i], "{}/predictions/target_patch_{}.tiff".format(gv.sn_model_path, k))
                ImageUtils.imsave(
                    s_target_patchs[i], "{}/predictions/s_target_patch_{}.tiff".format(gv.sn_model_path, k))
                # ImageUtils.imsave(
                #     prediction_patchs[i], "{}/predictions/prediction_patch_{}.tiff".format(gv.sn_model_path, k))
            
            print("score for pair {}: p:{}, n:{}".format(
                k, p[i][0], n[i][0]))

