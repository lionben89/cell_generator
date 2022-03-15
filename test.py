import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
from metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
import global_vars as gv
import os

tf.compat.v1.enable_eager_execution()

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))


test_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size=8,num_batches=32,patch_size=gv.patch_size,min_precentage=0.0,max_precentage=0.95)

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
        prediction = aae(patchs).numpy()
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

elif (gv.model_type == "AE"):
    from models.AE import *
    ae = keras.models.load_model(gv.model_path)

    patch_0 = test_dataset.__getitem__(0)[0]
    z = ae.encoder(patch_0)
    reconstruction_patch_0 = ae.decoder(z).numpy()
    for i in range(patch_0.shape[0]):
        ImageUtils.imsave(patch_0[i],"ae_original_patch_{}.tiff".format(i))
        ImageUtils.imsave(reconstruction_patch_0[i],"ae_reconstruction_patch_{}.tiff".format(i))

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
    from models.UNET import *
    unet = keras.models.load_model(gv.unet_model_path)
    if (not os.path.exists("{}/predictions".format(gv.unet_model_path))):
        os.makedirs("{}/predictions".format(gv.unet_model_path))
    n = test_dataset.__len__()
    ppc = 0
    out = 0
    for j in range(n):
        patchs = test_dataset.__getitem__(j)[0]
        target_patchs = test_dataset.__getitem__(j)[1]
        prediction = unet(patchs).numpy()
        for i in range(patchs.shape[0]):
            k = j*patchs.shape[0] +i
            if k <10:
                ImageUtils.imsave(patchs[i],"{}/predictions/input_patch_{}.tiff".format(gv.unet_model_path,k))
                ImageUtils.imsave(target_patchs[i],"{}/predictions/target_patch_{}.tiff".format(gv.unet_model_path,k))
                ImageUtils.imsave(prediction[i],"{}/predictions/prediction_patch_{}.tiff".format(gv.unet_model_path,k))
            p = pearson_corr(target_patchs[i],prediction[i])
            
            ppc+=p
            print("pearson correlation for image {}: {}".format(k,p))
    print("avg pearson correlation: {}".format(ppc/((patchs.shape[0]*n)-out)))
    
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
        z[:,35]=-2
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
    
    