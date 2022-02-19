import os
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
import global_vars as gv
from callbacks import *
import numpy as np
tf.compat.v1.enable_eager_execution()
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# CONTINUE_TRAINING = False
CONTINUE_TRAINING = True

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

train_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,16,patch_size=gv.patch_size)
# validation_dataset = DataGen(gv.test_ds_path,gv.input,gv.target,8,patch_size=gv.patch_size)


# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# train_dataset = np.expand_dims(np.pad(x_train/255.,(2,2))[2:-2],axis=-1)

if (gv.model_type == "VAE"):
    from models.VAE import *
    encoder, layer_dim, filters = get_encoder(gv.patch_size,gv.latent_dim)
    encoder.summary()
    decoder = get_decoder(gv.latent_dim,gv.patch_size,layer_dim,filters)
    decoder.summary()
    vae = VAE(encoder,decoder,beta=0.5) ## beta for reconstruction 1 for KL
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001))

    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        vae_pt = keras.models.load_model(gv.model_path)
        vae.set_weights(vae_pt.get_weights())
        
    checkpoint_callback = keras.callbacks.ModelCheckpoint(gv.model_path, save_best_only=True, save_weights_only=True, save_format="tf")
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20)
    callbacks = [checkpoint_callback,early_stop_callback]
    vae.fit(train_dataset, epochs=gv.number_epochs, callbacks=callbacks) ## validation_data=validation_dataset,
    vae.save(gv.model_path,save_format="tf")
    
elif (gv.model_type == "AAE"):
    from models.AAE import *
    encoder, layer_dim, filters = get_encoder(gv.patch_size,gv.latent_dim)
    encoder.summary()
    decoder = get_decoder(gv.latent_dim,gv.patch_size,layer_dim,filters)
    decoder.summary()
    discriminator_latent = get_discriminator_latent(gv.latent_dim)
    discriminator_latent.summary()
    discriminator_image = get_discriminator_image(gv.patch_size)
    discriminator_image.summary()
    aae = AAE(encoder,decoder,gv.patch_size,discriminator_latent,discriminator_image)
    aae.compile(keras.optimizers.Adam(learning_rate=0.00001), keras.optimizers.Adam(learning_rate=0.00001), keras.optimizers.Adam(learning_rate=0.0005))

    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        aae_pt = keras.models.load_model(gv.model_path)
        aae.set_weights(aae_pt.get_weights())  
    aae.fit(train_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(50,gv.number_epochs),aae)])
    
elif (gv.model_type == "AE"):
    from models.AE import *
    encoder, layer_dim, filters = get_encoder(gv.patch_size)
    encoder.summary()
    decoder = get_decoder(gv.patch_size,layer_dim,filters)
    decoder.summary()
    ae = AE(encoder,decoder)
    ae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        ae_pt = keras.models.load_model(gv.model_path)
        ae.set_weights(ae_pt.get_weights())
        
    ae.fit(train_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(20,ae)])    
    