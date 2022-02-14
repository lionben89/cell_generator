import tensorflow as tf
import tensorflow.keras as keras
from dataset import PatchDataGen
from models.VAE import *
from cell_imaging_utils.image.image_utils import ImageUtils
import matplotlib.pyplot as plt
import global_vars as gv

tf.compat.v1.enable_eager_execution()

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))


validation_dataset = PatchDataGen(gv.train_ds_path,"channel_signal","channel_signal",8,patch_size=gv.patch_size)

if (gv.model_type == "VAE"):
    from models.VAE import *
    vae = keras.models.load_model(gv.model_path)

    patch_0 = validation_dataset.__getitem__(0)[0]
    mu,sigma,sampled_z = vae.encoder(patch_0)
    sigma_zeros = tf.zeros(mu.shape)
    reconstruction_z = Sampling()([mu,sigma_zeros])
    reconstruction_patch_0 = vae.decoder(reconstruction_z).numpy()
    for i in range(patch_0.shape[0]):
        ImageUtils.imsave(patch_0[i],"original_patch_{}.tiff".format(i))
        ImageUtils.imsave(reconstruction_patch_0[i],"vae_reconstruction_patch_{}.tiff".format(i))

elif (gv.model_type == "AAE"):
    from models.AAE import *
    aae = keras.models.load_model(gv.model_path)

    patch_0 = validation_dataset.__getitem__(0)[0]
    reconstruction_patch_0 = aae.generator(patch_0).numpy()
    for i in range(patch_0.shape[0]):
        ImageUtils.imsave(patch_0[i],"original_patch_{}.tiff".format(i))
        ImageUtils.imsave(reconstruction_patch_0[i],"aae_reconstruction_patch_{}.tiff".format(i))