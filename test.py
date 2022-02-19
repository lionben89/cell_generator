import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
from models.VAE import *
from cell_imaging_utils.image.image_utils import ImageUtils
import global_vars as gv

tf.compat.v1.enable_eager_execution()

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))


validation_dataset = DataGen(gv.test_ds_path,gv.input,gv.target,8,num_batches=1,patch_size=gv.patch_size)

if (gv.model_type == "VAE"):
    from models.VAE import *
    vae = keras.models.load_model(gv.model_path)

    patch_0 = validation_dataset.__getitem__(0)[0]
    mu,sigma,sampled_z = vae.encoder(patch_0)
    sigma_zeros = tf.zeros(mu.shape)
    reconstruction_z = Sampling()([mu,sigma_zeros])
    reconstruction_patch_0 = vae.decoder(reconstruction_z).numpy()
    for i in range(patch_0.shape[0]):
        ImageUtils.imsave(patch_0[i],"vae_original_patch_{}.tiff".format(i))
        ImageUtils.imsave(reconstruction_patch_0[i],"vae_reconstruction_patch_{}.tiff".format(i))

elif (gv.model_type == "AAE"):
    from models.AAE import *
    aae = keras.models.load_model(gv.model_path)

    patchs = validation_dataset.__getitem__(0)[0]
    target_patchs = validation_dataset.__getitem__(0)[1]
    reconstruction_patch_0 = aae.generator(patchs).numpy() ##with sampling
    z_sample = sample_distribution(patchs.shape[0],gv.latent_dim) ##with out reference
    gen_patch = aae.decoder(z_sample).numpy()
    mu,sigma,sampled_z = aae.encoder(patchs)
    reconstruction_0 = aae.decoder(mu).numpy()
    for i in range(patchs.shape[0]):
        # ImageUtils.imsave(patchs[i],"input_patch_{}.tiff".format(i))
        ImageUtils.imsave(target_patchs[i],"target_patch_{}.tiff".format(i))
        ImageUtils.imsave(reconstruction_patch_0[i],"aae_reconstruction_with_sampling_patch_{}.tiff".format(i))
        ImageUtils.imsave(reconstruction_0[i],"aae_reconstruction_patch_{}.tiff".format(i))
        ImageUtils.imsave(gen_patch[i],"aae_generated_patch_{}.tiff".format(i))

elif (gv.model_type == "AE"):
    from models.AE import *
    ae = keras.models.load_model(gv.model_path)

    patch_0 = validation_dataset.__getitem__(0)[0]
    z = ae.encoder(patch_0)
    reconstruction_patch_0 = ae.decoder(z).numpy()
    for i in range(patch_0.shape[0]):
        ImageUtils.imsave(patch_0[i],"ae_original_patch_{}.tiff".format(i))
        ImageUtils.imsave(reconstruction_patch_0[i],"ae_reconstruction_patch_{}.tiff".format(i))