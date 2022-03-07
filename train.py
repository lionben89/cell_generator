import os
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
import global_vars as gv
from callbacks import *
from metrics import *

from models.Latent2Latent import Latent2Latent
from models.Latent2LatentRes import get_adaptor
from models.SampleGenerator import SampleGenerator
tf.compat.v1.enable_eager_execution()
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# CONTINUE_TRAINING = False
CONTINUE_TRAINING = True

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

train_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 64, patch_size=gv.patch_size,min_precentage=0,max_precentage=0.95)
# validation_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size = 4, num_batches = 2, patch_size=gv.patch_size,min_precentage=0.95,max_precentage=1)


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
    from models.AAE_patch import *
    encoder, layer_dim, filters = get_encoder(gv.patch_size,gv.latent_dim)
    encoder.summary()
    decoder = get_decoder(gv.latent_dim,gv.patch_size,layer_dim,filters)
    decoder.summary()
    discriminator_latent = get_discriminator_latent(gv.latent_dim)
    discriminator_latent.summary()
    discriminator_image = get_discriminator_image(gv.patch_size)
    discriminator_image.summary()
    aae = AAE(encoder,decoder,gv.patch_size,discriminator_latent,discriminator_image)
    aae.compile(d_latent_optimizer = keras.optimizers.Adam(learning_rate=0.0001), d_image_optimizer = keras.optimizers.Adam(learning_rate=0.0001), g_optimizer = keras.optimizers.Adam(learning_rate=0.0001))

    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        aae_pt = keras.models.load_model(gv.model_path)
        aae.set_weights(aae_pt.get_weights())  
    aae.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(10,gv.number_epochs),aae)])
    
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

elif (gv.model_type == "L2L"):
    from models.Latent2Latent import *
    aae = keras.models.load_model(gv.model_path)
    
    encoder, layer_dim, filters = get_encoder(gv.patch_size,gv.latent_dim)
    encoder.summary()

    l2l = Latent2Latent(encoder,aae.encoder,aae.decoder)
    l2l.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    if CONTINUE_TRAINING and os.path.exists(gv.latent_to_latent_model_path):
        l2l_pt = keras.models.load_model(gv.latent_to_latent_model_path)
        l2l.set_weights(l2l_pt.get_weights())  
    l2l.fit(train_dataset, validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[keras.callbacks.EarlyStopping(monitor="val_input_encoder_loss",patience=20,restore_best_weights=True),SaveModelCallback(min(20,gv.number_epochs),l2l,gv.latent_to_latent_model_path)])
    l2l.save(gv.latent_to_latent_model_path,save_format="tf")
    
elif (gv.model_type == "L2LRes"):
    from models.Latent2LatentRes import *
    aae = keras.models.load_model(gv.model_path)
    
    encoder, layer_dim, filters = get_encoder(gv.patch_size,gv.latent_dim)
    encoder.summary()
    
    adaptor = get_adaptor(gv.patch_size)
    adaptor.summary()
    
    l2lres = Latent2LatentRes(encoder,adaptor,aae.encoder,aae.decoder)
    l2lres.compile(encoder_optimizer=keras.optimizers.Adam(learning_rate=0.0001),adaptor_optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    if CONTINUE_TRAINING and os.path.exists(gv.latent_to_latent_model_path):
        l2l_pt = keras.models.load_model(gv.latent_to_latent_model_path)
        l2lres.set_weights(l2l_pt.get_weights())  
    l2lres.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[keras.callbacks.EarlyStopping(monitor="val_total_loss",patience=20,restore_best_weights=True),SaveModelCallback(min(20,gv.number_epochs),l2lres,gv.latent_to_latent_model_path)])

elif (gv.model_type == "UNET"):
    from models.UNET import *
    
    unet_model = get_unet(gv.patch_size)
    unet_model.summary()
    
    unet_model = UNET(unet_model)
    unet_model.compile(optimizer = keras.optimizers.Adam(learning_rate=1e-05))
    
    if CONTINUE_TRAINING and os.path.exists(gv.unet_model_path):
        unet_pt = keras.models.load_model(gv.unet_model_path)
        unet_model.set_weights(unet_pt.get_weights())  
    unet_model.fit(train_dataset, epochs=gv.number_epochs, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",patience=40,restore_best_weights=True),SaveModelCallback(min(20,gv.number_epochs),unet_model,gv.unet_model_path)])
    unet_model.save(gv.unet_model_path,save_format="tf")
    
elif (gv.model_type == "SG"):
    from models.SampleGenerator import *
    from models.UNET import *
    
    unet = keras.models.load_model(gv.unet_model_path)
    unet.summary()
    
    aae = keras.models.load_model(gv.model_path)
    aae.summary()
    
    # lp = get_latent_preprocessor(gv.latent_dim,gv.patch_size,(1,16,16,1))
    # lp.summary()
    
    lp = aae.decoder
    lp.trainable = False
    
    # adaptor = get_adaptor((*gv.patch_size[:-1],33))
    # adaptor.summary()
    adaptor = get_unet((*gv.patch_size[:-1],64))
    adaptor.summary()
    
    discriminator_image = get_discriminator_image(gv.patch_size)
    discriminator_image.summary()
    
    sg = SampleGenerator(gv.latent_dim, gv.patch_size, lp, adaptor, discriminator_image, unet, aae)
    sg.compile(d_optimizer = keras.optimizers.Adam(learning_rate=0.00001),g_optimizer = keras.optimizers.Adam(learning_rate=1e-04))
    
    if CONTINUE_TRAINING and os.path.exists(gv.sg_model_path):
        sg_pt = keras.models.load_model(gv.sg_model_path)
        sg.set_weights(sg_pt.get_weights())  
    sg.fit(train_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(5,gv.number_epochs),sg,gv.sg_model_path)])
    sg.save(gv.sg_model_path,save_format="tf")