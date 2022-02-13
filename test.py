import tensorflow as tf
import tensorflow.keras as keras
from dataset import PatchDataGen
from models.VAE import *
from cell_imaging_utils.image.image_utils import ImageUtils
import matplotlib.pyplot as plt
tf.compat.v1.enable_eager_execution()

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

patch_size =  (1,256,256,1)

validation_dataset = PatchDataGen("/sise/home/lionb/cell_generator/image_list_test.csv","channel_signal","channel_signal",8,patch_size=patch_size)

vae = keras.models.load_model("./vae_single_cell")

patch_0 = validation_dataset.__getitem__(0)[0]
mu,sigma,sampled_z = vae.encoder(patch_0)
sigma_zeros = tf.zeros(mu.shape)
reconstruction_z = Sampling()([mu,sigma_zeros])
reconstruction_patch_0 = vae.decoder(reconstruction_z).numpy()
for i in range(patch_0.shape[0]):
    ImageUtils.imsave(patch_0[i],"original_patch_{}.tiff".format(i))
    ImageUtils.imsave(reconstruction_patch_0[i],"reconstruction_patch_{}.tiff".format(i))
