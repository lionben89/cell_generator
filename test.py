import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
from metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
import global_vars as gv
import os
from utils import get_weights

#Methods to predict and compare to GT
gv.model_type = "UNET"
for_clf = (gv.model_type == "CLF")

predictors=None #True w_dna
gv.model_path = "../mg_model_ne_13_05_24_1.5_inst5"#../mg_model_unet_13_05_24_1.5"

#Input and target channels in the image
gv.input = "channel_signal"
gv.target = "channel_target"

#Organelle to predict the model upon
gv.organelle = "Nuclear-envelope" #"Tight-junctions" #Actin-filaments" #"Golgi" #"Microtubules" #"Endoplasmic-reticulum" 
#"Plasma-membrane" #"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)"

#Assemble the proper tarining csvs by the organelle, model type, and if the data is pertrubed or not
if gv.model_type == "CLF":
    gv.input = "channel_target"
    gv.target = "channel_target"
    gv.train_ds_path = "/home/lionb/cell_generator/image_list_train.csv"
    gv.test_ds_path = "/home/lionb/cell_generator/image_list_test.csv"
else:
    # gv.test_ds_path = "/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/{}/image_list_train.csv".format(gv.organelle)
    gv.test_ds_path = "/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/{}/image_list_test.csv".format(gv.organelle)
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

print("Model: ",gv.model_path)
print("Organelle: ",gv.organelle)
print("Compound: ",compound)
print("Vehicle: ", drug)

#Create a test dataset with 1 sample just to take the list of the images
test_dataset = DataGen(ds_path, gv.input, gv.target, batch_size=1, num_batches=1, patch_size=gv.patch_size, min_precentage=0, max_precentage=1, augment=False, norm_type=norm_type,predictors=predictors)

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
    clf = keras.models.load_model(gv.model_path)
    dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 4, num_batches = 8, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False, for_clf=for_clf)
    for i in range(20):
        clf.evaluate(dataset)
        
elif (gv.model_type == "UNET"):
    # images = [1]
    images = range(15,17)#range(min(10,test_dataset.df.get_shape()[0]))
    unet = keras.models.load_model(gv.model_path)
    if (not os.path.exists("{}/predictions".format(gv.model_path))):
        os.makedirs("{}/predictions".format(gv.model_path))
    pcc = 0
    for image_index in images:        
        # image_index = 1
        if (not os.path.exists("{}/predictions/{}".format(gv.model_path,image_index))):
            os.makedirs("{}/predictions/{}".format(gv.model_path,image_index))
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
                target_image = ImageUtils.normalize(target_image,max_value=1.0,dtype=np.float32)
                input_image = ImageUtils.normalize(input_image,max_value=1.0,dtype=np.float32)
                pred_image = ImageUtils.normalize(pred_image,max_value=1.0,dtype=np.float32)
            else:
                target_image = ImageUtils.normalize_std(target_image)
                input_image = ImageUtils.normalize_std(input_image)
                pred_image = ImageUtils.normalize_std(pred_image)
        
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
                    patch = ImageUtils.slice_image(input_image,s)
                    pred_patch = ImageUtils.slice_image(pred_image,s)
                    patch = ImageUtils.to_shape(patch,gv.patch_size,min_shape=gv.patch_size)
                    if predictors:
                        pred_patch = ImageUtils.to_shape(pred_patch,gv.patch_size,min_shape=gv.patch_size)
                        
                        # patch_p = unet(np.expand_dims([patch],axis=0))
                        patch_p = unet.unet(np.expand_dims(np.concatenate([patch,pred_patch],axis=-1),axis=0))
                    else:
                        patch_p = unet(np.expand_dims(patch,axis=0))
                    weights = get_weights(patch_p.shape)
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
        
        ImageUtils.imsave(input_cut.astype(np.float16),"{}/predictions/{}/input_patch_{}.tiff".format(gv.model_path,image_index,image_index))
        ImageUtils.imsave(target_cut.astype(np.float16),"{}/predictions/{}/target_patch_{}.tiff".format(gv.model_path,image_index,image_index))
        ImageUtils.imsave(prediction_cut.astype(np.float16),"{}/predictions/{}/prediction_patch_{}.tiff".format(gv.model_path,image_index,image_index))
        # ImageUtils.imsave(nuc_seg_cut.astype(np.float16),"{}/predictions/{}/nuc_seg_patch_{}.tiff".format(gv.model_path,image_index,image_index))
        p=pearson_corr(target_cut, prediction_cut)
        pcc+=p
        print("pearson corr for image:{} is :{}".format(image_index,p))
  
        print("prediction - mean:{}, std:{}".format(np.mean(prediction_cut,dtype=np.float64),np.std(prediction_cut,dtype=np.float64)))
        print("target - mean:{}, std:{}".format(np.mean(target_image,dtype=np.float64),np.std(target_image,dtype=np.float64)))        
    print("average pcc:{}".format(pcc/len(images)))

elif (gv.model_type == "MG"):
    from mg_analyzer import analyze_th
    analyze_th(test_dataset,"regular",mask_image=None,manual_th="full",save_image=1,save_histo=False,weighted_pcc = False, model_path=gv.model_path,model=None,compound=None,images=[0,1])

elif (gv.model_type == "RC"):
    rc = keras.models.load_model(gv.model_path)
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
    pm = keras.models.load_model(gv.model_path)
    if (not os.path.exists("{}/predictions".format(gv.pm_model_path))):
        os.makedirs("{}/predictions".format(gv.pm_model_path))
    unet = keras.models.load_model(gv.interpert_model_path)
    len = test_dataset.__len__()
    total_ba = []
    for j in range(len):
        patchs = test_dataset.__getitem__(j)[0]
        labels = test_dataset.__getitem__(j)[1]
        predictions = pm([patchs[0],patchs[1]]).numpy()
        total_ba.append(tf.reduce_mean(keras.metrics.binary_accuracy(labels,predictions)))
        for i in range(patchs[0].shape[0]):
            k = j*patchs[0].shape[0] + i
            if k < 32:
                ImageUtils.imsave(
                    patchs[0][i], "{}/predictions/input_patch_{}.tiff".format(gv.pm_model_path, k))
                ImageUtils.imsave(
                    patchs[1][i], "{}/predictions/target_patch_{}.tiff".format(gv.pm_model_path, k))
            print("score for pair {}: prediction:{}, label:{}".format(
                k, predictions[i], labels[i]))
    print("total accuracy is :",np.average(total_ba))
    

