import tensorflow as tf
import tensorflow.keras as keras
from dataset import PatchDataGen
import global_vars as gv
from models.AAE import get_discriminator
tf.compat.v1.enable_eager_execution()

# CONTINUE_TRAINING = False
CONTINUE_TRAINING = True

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

train_dataset = PatchDataGen(gv.train_ds_path,"channel_signal","channel_signal",128,patch_size=gv.patch_size)
validation_dataset = PatchDataGen(gv.test_ds_path,"channel_signal","channel_signal",8,patch_size=gv.patch_size)
if (gv.model_type == "VAE"):
    from models.VAE import *
    encoder, layer_dim, filters = get_encoder(gv.patch_size,gv.latent_dim)
    encoder.summary()
    decoder = get_decoder(gv.latent_dim,gv.patch_size,layer_dim,filters)
    decoder.summary()
    vae = VAE(encoder,decoder,beta=0.5) ## beta for reconstruction 1 for KL
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    if CONTINUE_TRAINING:
        vae_pt = keras.models.load_model(gv.model_path)
        vae.set_weights(vae_pt.get_weights())
        
    checkpoint_callback = keras.callbacks.ModelCheckpoint(gv.model_path, save_best_only=True, save_weights_only=True, save_format="tf")
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20)
    callbacks = [checkpoint_callback,early_stop_callback]
    vae.fit(train_dataset,validation_data=validation_dataset, epochs=100, callbacks=callbacks) ## validation_data=validation_dataset,
    vae.save(gv.model_path,save_format="tf")
    
elif (gv.model_type == "AAE"):
    from models.AAE import *
    encoder, layer_dim, filters = get_encoder(gv.patch_size,gv.latent_dim)
    encoder.summary()
    decoder = get_decoder(gv.latent_dim,gv.patch_size,layer_dim,filters)
    decoder.summary()
    discriminator = get_discriminator(gv.latent_dim)
    discriminator.summary()
    aae = AAE(encoder,decoder,gv.patch_size,discriminator)
    aae.compile(keras.optimizers.Adam(learning_rate=0.00001), keras.optimizers.Adam(learning_rate=0.0001))

    if CONTINUE_TRAINING:
        aae_pt = keras.models.load_model(gv.model_path)
        aae.set_weights(aae_pt.get_weights())
    aae.fit(train_dataset, epochs=1000)
    aae.save(gv.model_path,save_format="tf")