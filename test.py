import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
from metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
import global_vars as gv
import os

tf.compat.v1.enable_eager_execution()

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))


test_dataset = DataGen(gv.test_ds_path,gv.input,gv.target,batch_size=4,num_batches=32,patch_size=gv.patch_size,min_precentage=0.0,max_precentage=0.95,augment=False)

def slice_image(image_ndarray:np.ndarray, indexes:list)->np.ndarray:
    n_dim = len(image_ndarray.shape)
    slices = [slice(None)] * n_dim
    for i in range(len(indexes)):
        slices[i] = slice(indexes[i][0],indexes[i][1])
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
            k = j*patchs.shape[0] +i
            if k <10:
                ImageUtils.imsave(patchs[i],"{}/predictions/input_patch_{}.tiff".format(gv.model_path,k))
                ImageUtils.imsave(target_patchs[i],"{}/predictions/target_patch_{}.tiff".format(gv.model_path,k))
                ImageUtils.imsave(prediction[i],"{}/predictions/prediction_patch_{}.tiff".format(gv.model_path,k))
            p = pearson_corr(target_patchs[i],prediction[i])
            
            ppc+=p
            print("pearson correlation for image {}: {}".format(k,p))
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
        z[:,56]=-2
        alteredm2 = aae.decoder(z).numpy()
        z[:,56]=2
        alteredp2 = aae.decoder(z).numpy()
        for i in range(patchs.shape[0]):    
            k = j*patchs.shape[0] +i
            if k <10:
                ImageUtils.imsave(alteredm2[i]*255,"{}/predictions/altered_minus2_patch_{}.tiff".format(gv.model_path,k))
                ImageUtils.imsave(alteredp2[i]*255,"{}/predictions/altered_plus2_patch_{}.tiff".format(gv.model_path,k))
                ImageUtils.imsave(target_patchs[i]*255,"{}/predictions/target_patch_{}.tiff".format(gv.model_path,k))
                ImageUtils.imsave(prediction[i]*255,"{}/predictions/prediction_patch_{}.tiff".format(gv.model_path,k))
            p = PSNR(target_patchs[i]*255,prediction[i]*255)
            
            ppc+=p
            print("psnr correlation for image {}: {}".format(k,p))
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
            k = j*patchs.shape[0] +i
            if k <10:
                ImageUtils.imsave(patchs[i],"{}/predictions/input_patch_{}.tiff".format(gv.model_path,k))
                ImageUtils.imsave(target_patchs[i],"{}/predictions/target_patch_{}.tiff".format(gv.model_path,k))
                ImageUtils.imsave(prediction[i],"{}/predictions/prediction_patch_{}.tiff".format(gv.model_path,k))
            p = pearson_corr(target_patchs[i],prediction[i])
            
            ppc+=p
            print("pearson correlation for image {}: {}".format(k,p))
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
        ImageUtils.imsave(patchs[i],"input_patch_{}.tiff".format(i))
        ImageUtils.imsave(target_patchs[i],"target_patch_{}.tiff".format(i))
        ImageUtils.imsave(prediction[i],"l2l_prediction_patch_{}.tiff".format(i))
        print("latent dim diff for image {}: {}".format(i,sum(np.abs(z_true[i]-z[i]))))
        print("pearson correlation for image {}: {}".format(i,pearson_corr(target_patchs[i],prediction[i])))

elif (gv.model_type == "L2LRes"):
    from models.Latent2LatentRes import *
    l2l = keras.models.load_model(gv.latent_to_latent_model_path)

    patchs = test_dataset.__getitem__(0)[0]
    target_patchs = test_dataset.__getitem__(0)[1]
    prediction = l2l(patchs)

    for i in range(patchs.shape[0]):
        ImageUtils.imsave(patchs[i],"input_patch_{}.tiff".format(i))
        ImageUtils.imsave(target_patchs[i],"target_patch_{}.tiff".format(i))
        ImageUtils.imsave(prediction[i],"l2lres_prediction_patch_{}.tiff".format(i))
        print("pearson correlation for image {}: {}".format(i,pearson_corr(target_patchs[i],prediction[i])))

elif (gv.model_type == "UNET"):
    unet = keras.models.load_model(gv.unet_model_path)
    if (not os.path.exists("{}/predictions".format(gv.unet_model_path))):
        os.makedirs("{}/predictions".format(gv.unet_model_path))
    n = test_dataset.__len__()
    ppc = 0
    out = 0
    for j in range(n):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]
        prediction = (unet(patchs)).numpy()
        # prediction = unet.unet(patchs).numpy()
        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] +i
            if k <10:
                ImageUtils.imsave(patchs[i],"{}/predictions/input_patch_{}.tiff".format(gv.unet_model_path,k))
                ImageUtils.imsave(target_patchs[i],"{}/predictions/target_patch_{}.tiff".format(gv.unet_model_path,k))
                ImageUtils.imsave(prediction[i],"{}/predictions/prediction_patch_{}.tiff".format(gv.unet_model_path,k))
            p = dice(target_patchs[i],np.round(prediction[i]))
            
            ppc+=p
            print("dice correlation for image {}: {}".format(k,p))
    print("avg dice correlation: {}".format(ppc/((patchs.shape[0]*n)-out)))

    # image_index = 1
    # image_path = test_dataset.df.get_item(image_index,'path_tiff')
    # input_image, input_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.input_col)
    # target_image, target_new_file_path = test_dataset.get_image_from_ssd(image_path,test_dataset.target_col)
    # if (input_image is None or target_image is None):
    #     image_ndarray = None
    #     image_ndarray = ImageUtils.image_to_ndarray(ImageUtils.imread(image_path))
    #     channel_index = int(test_dataset.df.get_item(image_index,test_dataset.input_col))
    #     input_image = ImageUtils.get_channel(image_ndarray,channel_index)    
    #     channel_index = int(test_dataset.df.get_item(image_index,test_dataset.target_col))
    #     target_image = ImageUtils.get_channel(image_ndarray,channel_index)  
    # i=0
    # j=0
    # k=0
    # prediction = np.zeros_like(target_image)
    # d = np.zeros_like(target_image)
    # o = 256
    # while i<=input_image.shape[0]-gv.patch_size[0]:
    #     while j<=input_image.shape[1]-gv.patch_size[1]:
    #         while k<=input_image.shape[2]-gv.patch_size[2]:
    #             s = [(i,i+gv.patch_size[0]),(j,j+gv.patch_size[1]),(k,k+gv.patch_size[2])]
    #             patch = slice_image(input_image,s)
    #             patch = ImageUtils.to_shape(patch,gv.patch_size,min_shape=gv.patch_size)
    #             patch_p = unet(np.expand_dims(patch,axis=0))
    #             prediction[i:i+gv.patch_size[0],j:j+gv.patch_size[1],k:k+gv.patch_size[2]] += patch_p
    #             d[i:i+gv.patch_size[0],j:j+gv.patch_size[1],k:k+gv.patch_size[2]] += 1
    #             k+=o
    #         k=0
    #         j+=o
    #     j=0
    #     i+=16
    # ImageUtils.imsave(input_image,"{}/predictions/input_patch_{}.tiff".format(gv.unet_model_path,image_index))
    # ImageUtils.imsave(target_image,"{}/predictions/target_patch_{}.tiff".format(gv.unet_model_path,image_index))
    # ImageUtils.imsave(prediction/(d+0.00001),"{}/predictions/prediction_patch_{}.tiff".format(gv.unet_model_path,image_index))           
    
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
        z[:,56]=-2
        target_aae = sg.aae.decoder(z).numpy()
        prediction =  sg.generator([patchs,z]).numpy()
        prediction_unet = sg.unet(prediction).numpy()
        
        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] +i
            if k <10:
                ImageUtils.imsave(patchs[i],"{}/predictions/input_patch_{}.tiff".format(gv.sg_model_path,k))
                ImageUtils.imsave(target_patchs[i],"{}/predictions/target_patch_{}.tiff".format(gv.sg_model_path,k))
                ImageUtils.imsave(prediction[i],"{}/predictions/adapted_input_patch_{}.tiff".format(gv.sg_model_path,k))
                ImageUtils.imsave(target_aae[i],"{}/predictions/adapted_output_patch_{}.tiff".format(gv.sg_model_path,k))
                ImageUtils.imsave(prediction_unet[i],"{}/predictions/predicted_adapted_output_patch_{}.tiff".format(gv.sg_model_path,k))
                ImageUtils.imsave(target_unet[i],"{}/predictions/target_unet_output_patch_{}.tiff".format(gv.sg_model_path,k))
            p = pearson_corr(target_aae[i],prediction_unet[i])
            
            ppc+=p
            print("pearson correlation for image {}: {}".format(k,p))
    print("avg pearson correlation: {}".format(ppc/((patchs.shape[0]*n)-out)))
    
    