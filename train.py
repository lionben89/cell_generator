import os
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
import global_vars as gv
from callbacks import *
from metrics import *
from models.MaskGenerator import MaskGenerator
from models.RegCNN import RegCNN, get_reg
from models.ShuffleGenerator import ShuffleGenerator
from models.ZeroGenerator import ZeroGenerator

# tf.config.run_functions_eagerly(True)
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# CONTINUE_TRAINING = False
CONTINUE_TRAINING = True

gv.model_type = "UNET"
for_clf = (gv.model_type == "CLF")
gv.unet_model_path = "./unet_model_22_05_22_membrane_w_dna"
gv.mg_model_path = "mg_model_tj_10_06_22_5_0_new"
gv.clf_model_path = "./clf_model_14_12_22-1"
gv.organelle = "Plasma-membrane" #"Tight-junctions" #Actin-filaments" #"Golgi" #"Microtubules" #"Endoplasmic-reticulum" 
#"Plasma-membrane" #"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)"
if gv.model_type == "CLF":
    gv.input = "channel_target"
    gv.target = "channel_target"
    gv.train_ds_path = "/home/lionb/cell_generator/image_list_train.csv"
    gv.test_ds_path = "/home/lionb/cell_generator/image_list_test.csv"
else:
    gv.train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(gv.organelle)
    gv.test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_test.csv".format(gv.organelle)
gv.batch_size = 4 #4
noise_scale = 1.0
norm_type = "std"
gv.patch_size = (32,128,128,1)

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
train_dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 16, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=0.8,augment=True,norm_type=norm_type, for_clf=for_clf, predictors=True) #predictors={"Nuclear-envelope":ne_unet,"Nucleolus-(Granular-Component)":ngc_unet}
validation_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 4, patch_size=gv.patch_size,min_precentage=0.8,max_precentage=1.0,augment=False,norm_type=norm_type,for_clf=for_clf,predictors=True) #,predictors={"Nuclear-envelope":ne_unet,"Nucleolus-(Granular-Component)":ngc_unet})


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
    unet_model = get_unet((gv.patch_size[0],gv.patch_size[1],gv.patch_size[2],2),activation="linear")
    unet_model.summary()
    
    unet_model = UNET(unet_model)
    unet_model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001))
    # unet_model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),loss=keras.losses.MeanSquaredError())
    
    if CONTINUE_TRAINING and os.path.exists(gv.unet_model_path):
        unet_pt = keras.models.load_model(gv.unet_model_path)                           
        unet_model.set_weights(unet_pt.get_weights())  
    checkpoint_callback = SaveModelCallback(min(1,gv.number_epochs),unet_model,gv.unet_model_path)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=10,monitor="val_loss",restore_best_weights=True)
    unet_model.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[checkpoint_callback,early_stop_callback])
    unet_model.save(gv.unet_model_path,save_format="tf")

elif (gv.model_type == "CLF"):
    from models.Classifier import get_clf
    
    clf_model  = get_clf(gv.patch_size, num_classes=13)
    checkpoint_callback = SaveModelCallback(min(5,gv.number_epochs),clf_model,gv.clf_model_path,monitor="val_loss",save_all=True)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=10,monitor="val_loss",restore_best_weights=True)
    clf_model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),loss=keras.losses.SparseCategoricalCrossentropy(),metrics=[keras.metrics.SparseCategoricalAccuracy()])
    clf_model.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[checkpoint_callback,early_stop_callback])
    clf_model.save(gv.clf_model_path,save_format="tf")
    
elif (gv.model_type == "UNETP"):
    from models.UNETO_perception import *
    
    unet_model = get_unet(gv.patch_size,activation="relu")
    unet_model.summary()
    
    pl_model,pl_preprocess = get_perception_model(gv.patch_size)
    unet_model = UNET(unet_model,pl_model,pl_preprocess,gv.patch_size)
    unet_model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),run_eagerly=False)
    
    if CONTINUE_TRAINING and os.path.exists(gv.unet_model_path):
        unet_pt = keras.models.load_model(gv.unet_model_path)                           
        unet_model.set_weights(unet_pt.get_weights())  
    checkpoint_callback = SaveModelCallback(min(1,gv.number_epochs),unet_model,gv.unet_model_path,monitor="val_total_loss")
    early_stop_callback = keras.callbacks.EarlyStopping(patience=300,monitor="val_total_loss",restore_best_weights=True)
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

elif (gv.model_type == "ShG" or gv.model_type == "ShGD"):
    from models.InputGenerator import *
    from models.UNETO import *
    
    unet = keras.models.load_model(gv.unet_model_path)
    unet.summary()
    

    adaptor = get_unet((*gv.patch_size[:-1],64),activation="linear")
    adaptor.summary()
    
    discriminator_image = get_discriminator_image(gv.patch_size)
    discriminator_image.summary()
    
    shg = InputGenerator(gv.patch_size,adaptor, discriminator_image, unet)
    shg.unet.trainable = False
    
    shg.compile(d_optimizer = keras.optimizers.Adam(learning_rate=0.00001),g_optimizer = keras.optimizers.Adam(learning_rate=0.0001))
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True,monitor="val_total_loss")
    checkpoint_callback = SaveModelCallback(min(1,gv.number_epochs),shg,gv.shg_model_path,monitor="val_total_loss")
    if CONTINUE_TRAINING and os.path.exists(gv.shg_model_path):
        shg_pt = keras.models.load_model(gv.shg_model_path)
        shg.set_weights(shg_pt.get_weights())  
    shg.fit(train_dataset, validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[checkpoint_callback,early_stop_callback])
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
    
    # adaptor = get_adaptor((*gv.patch_size[:-1],32))
    # adaptor.summary()
    
    adaptor = get_unet((*gv.patch_size[:-1],64),activation="sigmoid") 
    adaptor.summary()
    
    mg = MaskGenerator(gv.patch_size, adaptor, unet)
    mg.unet.trainable = False
    
    
    
    # checkpoint_callback = SaveModelCallback(min(1,gv.number_epochs),mg,gv.mg_model_path,monitor="val_stop",term="val_pcc",term_value=0.92)
    if CONTINUE_TRAINING and os.path.exists(gv.mg_model_path):
        mg_pt = keras.models.load_model(gv.mg_model_path)
        mg.set_weights(mg_pt.get_weights())
    
    
    for j in range(5):
        mask_loss_weight=0.1
        checkpoint_callback = SaveModelCallback(min(1,gv.number_epochs),mg,gv.mg_model_path,monitor="val_stop",term="val_pcc",term_value=0.88)
        for i in range(10):
            print("mask_loss_weight: ",mask_loss_weight)
            print("noise_scale: ",noise_scale*(j+1))
            early_stop_callback = keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True,monitor="val_stop")
            mg.compile(g_optimizer = keras.optimizers.Adam(learning_rate=0.0001),mask_loss_weight=mask_loss_weight,mask_size_loss_weight=mask_loss_weight,run_eagerly=False,noise_scale=noise_scale*(j+1))
            mg.fit(train_dataset, validation_data=validation_dataset, epochs=100, callbacks=[checkpoint_callback,early_stop_callback]) 
            mask_loss_weight = mask_loss_weight+0.1
        
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
    
    pm = get_pm((16,64,64,1))
    pm.summary()
    
    # pm = PMCNN(pm)
    
    pm.compile(optimizer = keras.optimizers.Adam(learning_rate=0.000004),loss=keras.losses.binary_crossentropy ,metrics=[keras.metrics.BinaryAccuracy(),keras.metrics.Precision(),keras.metrics.Recall()]) #
    early_stop_callback = keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True,monitor="val_loss")
    if CONTINUE_TRAINING and os.path.exists(gv.pm_model_path):
        pm_pt = keras.models.load_model(gv.pm_model_path)
        pm.set_weights(pm_pt.get_weights())  
    FINE_TUNE = False
    if FINE_TUNE:
        for layer in pm.layers:
            if layer.name != "pair_matching_denseout" and layer.name != "pair_matching_dense2":
                layer.trainable = False
        
    train_dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 32, patch_size=gv.patch_size,min_precentage=0,max_precentage=0.9,augment=True,pairs=True,neg_ratio=1,cutoff=0.02,masking_pair=True)
    validation_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 8, patch_size=gv.patch_size,min_precentage=0.9,max_precentage=1,augment=False,pairs=True,neg_ratio=1,cutoff=0.02,masking_pair=True)
    pm.fit(train_dataset, validation_data=validation_dataset, epochs=1000 ,callbacks=[SaveModelCallback(min(3,gv.number_epochs),pm,gv.pm_model_path),early_stop_callback])
    pm.save(gv.pm_model_path,save_format="tf")

elif (gv.model_type == "SN"):
    from models.SN import *
    
    pm1 = get_pm((16,64,64,1),name="pm1")
    pm2 = get_pm((16,64,64,1),name="pm2")
    pm1.summary()
    
    sn = SN((16,64,64,1),pm1,pm2)
    
    sn.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001))
    # early_stop_callback = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True,monitor="val_loss")
    if CONTINUE_TRAINING and os.path.exists(gv.sn_model_path):
        sn_pt = keras.models.load_model(gv.sn_model_path)
        sn.set_weights(sn_pt.get_weights())  
    for i in range(0,1):
        train_dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 64, patch_size=gv.patch_size,min_precentage=0,max_precentage=0.9,augment=True,pairs=True,neg_ratio=i+1)
        validation_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 4, patch_size=gv.patch_size,min_precentage=0.9,max_precentage=1,augment=False,pairs=True,neg_ratio=i+1)
        sn.fit(train_dataset, validation_data=validation_dataset, epochs=100*(i+1)*(i+1), callbacks=[SaveModelCallback(min(1,gv.number_epochs),sn,gv.sn_model_path)])
