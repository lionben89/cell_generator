import os
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
import global_vars as gv
from callbacks import *
from metrics import *
from models.MaskGenerator import MaskGenerator
from models.PMCNN import PMCNN
from models.RegCNN import RegCNN, get_reg
from models.ShuffleGenerator import ShuffleGenerator
from models.ZeroGenerator import ZeroGenerator

tf.compat.v1.enable_eager_execution()
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# CONTINUE_TRAINING = False
CONTINUE_TRAINING = True

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
# ne_unet = keras.models.load_model("./unet_model_x_mse_2_3_nobn_ne")
# ngc_unet = keras.models.load_model("./unet_model_x_mse_2_3_nobn")
train_dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 80, patch_size=gv.patch_size,min_precentage=0,max_precentage=0.9,augment=True) #predictors={"Nuclear-envelope":ne_unet,"Nucleolus-(Granular-Component)":ngc_unet}
validation_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 16, patch_size=gv.patch_size,min_precentage=0.9,max_precentage=1,augment=False) #,predictors={"Nuclear-envelope":ne_unet,"Nucleolus-(Granular-Component)":ngc_unet})


# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# train_dataset = np.expand_dims(np.pad(x_train/255.,(2,2))[2:-2],axis=-1)

if (gv.model_type == "VAE"):
    from models.VAE import *
    encoder, layer_dim, filters = get_encoder(gv.patch_size,gv.latent_dim)
    encoder.summary()
    decoder = get_decoder(gv.latent_dim,gv.patch_size,layer_dim,filters)
    decoder.summary()
    vae = VAE(encoder,decoder,beta=1) ## beta for reconstruction 1 for KL
    
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001))

    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        vae_pt = keras.models.load_model(gv.model_path)
        vae.set_weights(vae_pt.get_weights())
        
    checkpoint_callback = SaveModelCallback(min(5,gv.number_epochs),vae)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20)
    callbacks = [checkpoint_callback,early_stop_callback]
    vae.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=callbacks) ## validation_data=validation_dataset,
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
    aae.compile(d_latent_optimizer = keras.optimizers.Adam(learning_rate=0.000005), d_image_optimizer = keras.optimizers.Adam(learning_rate=0.000005), g_optimizer = keras.optimizers.Adam(learning_rate=0.00002))

    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        aae_pt = keras.models.load_model(gv.model_path)
        aae.set_weights(aae_pt.get_weights())  
    aae.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(5,gv.number_epochs),aae)])
    
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
        
    ae.fit(train_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(5,ae)])    

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
    from models.UNETO import *
    
    unet_model = get_unet(gv.patch_size,activation="relu")
    unet_model.summary()
    
    unet_model = UNET(unet_model)
    # unet_model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),loss=keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=0.1)) #,metrics=keras.metrics.MeanIoU(2)
    unet_model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001))
    
    if CONTINUE_TRAINING and os.path.exists(gv.unet_model_path):
        unet_pt = keras.models.load_model(gv.unet_model_path)                           
        unet_model.set_weights(unet_pt.get_weights())  
    checkpoint_callback = SaveModelCallback(min(5,gv.number_epochs),unet_model,gv.unet_model_path)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20)
    unet_model.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[checkpoint_callback,early_stop_callback])
    unet_model.save(gv.unet_model_path,save_format="tf")

elif (gv.model_type == "UNET_seg"):
    from models.UNETO_seg import *
    
    pre_unet = keras.models.load_model(gv.pre_unet_model_path)
    
    unet_model = get_unet(gv.patch_size,activation="relu")
    unet_model.summary()
    
    unet_model = UNET(unet_model,pre_unet)
    # unet_model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),loss=keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=0.1)) #,metrics=keras.metrics.MeanIoU(2)
    unet_model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001))
    
    if CONTINUE_TRAINING and os.path.exists(gv.unet_model_path):
        unet_pt = keras.models.load_model(gv.unet_model_path)
        unet_model.set_weights(unet_pt.get_weights())  
    checkpoint_callback = SaveModelCallback(min(5,gv.number_epochs),unet_model,gv.unet_model_path)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=100)
    unet_model.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[checkpoint_callback,early_stop_callback])
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
    adaptor = get_unet((*gv.patch_size[:-1],64),activation="relu")
    adaptor.summary()
    
    discriminator_image = get_discriminator_image(gv.patch_size)
    discriminator_image.summary()
    
    sg = SampleGenerator(gv.latent_dim, gv.patch_size, lp, adaptor, discriminator_image, unet, aae)
    
    sg.compile(d_optimizer = keras.optimizers.Adam(learning_rate=0.00001),g_optimizer = keras.optimizers.Adam(learning_rate=0.0001))
    if CONTINUE_TRAINING and os.path.exists(gv.sg_model_path):
        sg_pt = keras.models.load_model(gv.sg_model_path)
        sg.set_weights(sg_pt.get_weights())  
    sg.fit(train_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(5,gv.number_epochs),sg,gv.sg_model_path)])
    sg.save(gv.sg_model_path,save_format="tf")

elif (gv.model_type == "ShG"):
    from models.ShuffleGenerator import *
    from models.UNETO import *
    
    unet = keras.models.load_model(gv.unet_model_path)
    unet.summary()
    

    adaptor = get_unet((*gv.patch_size[:-1],64),activation="relu")
    adaptor.summary()
    
    discriminator_image = get_discriminator_image(gv.patch_size)
    discriminator_image.summary()
    
    shg = ShuffleGenerator(gv.patch_size, adaptor, discriminator_image, unet)
    shg.unet.trainable = True
    
    shg.compile(d_optimizer = keras.optimizers.Adam(learning_rate=0.00001),g_optimizer = keras.optimizers.Adam(learning_rate=0.0001))
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True,monitor="val_total loss")
    if CONTINUE_TRAINING and os.path.exists(gv.shg_model_path):
        shg_pt = keras.models.load_model(gv.shg_model_path)
        shg.set_weights(shg_pt.get_weights())  
    shg.fit(train_dataset, validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(5,gv.number_epochs),shg,gv.shg_model_path),early_stop_callback])
    shg.save(gv.shg_model_path,save_format="tf")
    
elif (gv.model_type == "ZG"):
    from models.ZeroGenerator import *
    from models.UNETO import *
    
    unet = keras.models.load_model(gv.unet_model_path)
    unet.summary()
    
    # adaptor = get_adaptor((*gv.patch_size[:-1],32))
    # adaptor.summary()
    adaptor = get_unet((*gv.patch_size[:-1],32),activation="relu")
    adaptor.summary()
    
    zg = ZeroGenerator(gv.patch_size, adaptor, unet)
    zg.unet.trainable = False
    
    zg.compile(g_optimizer = keras.optimizers.Adam(learning_rate=0.0001))
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True,monitor="val_total loss")
    if CONTINUE_TRAINING and os.path.exists(gv.zg_model_path):
        zg_pt = keras.models.load_model(gv.zg_model_path)
        zg.set_weights(zg_pt.get_weights())  
    zg.fit(train_dataset, validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(1,gv.number_epochs),zg,gv.zg_model_path),early_stop_callback])
    zg.save(gv.zg_model_path,save_format="tf")

elif (gv.model_type == "MG"):
    from models.MaskGenerator import *
    from models.UNETO import *
    
    unet = keras.models.load_model(gv.unet_model_path)
    unet.summary()
    
    adaptor = get_adaptor((*gv.patch_size[:-1],32))
    adaptor.summary()
    # adaptor = get_unet((*gv.patch_size[:-1],32),activation="sigmoid")
    # adaptor.summary()
    
    mg = MaskGenerator(gv.patch_size, adaptor, unet)
    mg.unet.trainable = False
    
    mg.compile(g_optimizer = keras.optimizers.Adam(learning_rate=0.0005))
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True,monitor="val_total loss")
    if CONTINUE_TRAINING and os.path.exists(gv.mg_model_path):
        mg_pt = keras.models.load_model(gv.mg_model_path)
        mg.set_weights(mg_pt.get_weights())  
    mg.fit(train_dataset, validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(1,gv.number_epochs),mg,gv.mg_model_path),early_stop_callback])
    mg.save(gv.mg_model_path,save_format="tf")
        
elif (gv.model_type == "EAM"): #Explainable Adverserial Model
    from models.EAM import *
    from models.UNETO import *
    
    unet = keras.models.load_model(gv.unet_model_path)
    unet.summary()
    
    adaptor = get_adaptor(gv.patch_size)
    adaptor.summary()
    # adaptor = get_unet(gv.patch_size,activation="relu")
    # adaptor.summary()
    discriminator_image = get_discriminator_image(gv.patch_size)
    
    eam = EAM(gv.patch_size,adaptor, unet,discriminator_image)
    
    eam.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),d_optimizer = keras.optimizers.Adam(learning_rate=0.00001))
    if CONTINUE_TRAINING and os.path.exists(gv.eam_model_path):
        eam_pt = keras.models.load_model(gv.eam_model_path)
        eam.set_weights(eam_pt.get_weights())  
    early_stop_callback = keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True,monitor="val_total_loss")
    eam.fit(train_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(5,gv.number_epochs),eam,gv.eam_model_path),early_stop_callback]) #validation_data=validation_dataset
    eam.save(gv.eam_model_path,save_format="tf")
    
elif (gv.model_type == "RC"):
    from models.RegCNN import *
    
    regressor = get_reg(gv.patch_size)
    regressor.summary()
    
    rc = RegCNN(regressor)
    
    rc.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00001))
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True,monitor="val_loss")
    if CONTINUE_TRAINING and os.path.exists(gv.rc_model_path):
        rc_pt = keras.models.load_model(gv.rc_model_path)
        rc.set_weights(rc_pt.get_weights())  
    rc.fit(train_dataset, validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(1,gv.number_epochs),rc,gv.rc_model_path),early_stop_callback])
    rc.save(gv.rc_model_path,save_format="tf")
    
elif (gv.model_type == "PM"):
    from models.PMCNN import *
    
    pm = get_pm(gv.patch_size)
    pm.summary()
    
    pm = PMCNN(pm)
    
    pm.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001))
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True,monitor="val_loss")
    if CONTINUE_TRAINING and os.path.exists(gv.pm_model_path):
        pm_pt = keras.models.load_model(gv.pm_model_path)
        pm.set_weights(pm_pt.get_weights())  
    pm.fit(train_dataset, validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(5,gv.number_epochs),pm,gv.pm_model_path),early_stop_callback])
    pm.save(gv.pm_model_path,save_format="tf")
