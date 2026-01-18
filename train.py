import os
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
import global_vars as gv
from callbacks import *
from metrics import *
import pandas as pd
from utils import *

CONTINUE_TRAINING = False ## False to override current model with the same name

#Model to train
gv.model_type = "MG"
for_clf = (gv.model_type == "CLF")

#If Mask Interpreter then add the path to the model you want to interpret
gv.interpert_model_path = "../unet_model_22_05_22_mito_128" ## UNET model if in MG mode it is the model that we want to interpret
#path to the model
gv.model_path = "../mg_model_mito_13_05_24_noise_1.5_sim_0.0_target_6.0_mask_1.0_mse" ## the model will be saved here

#Input and target channels in the image
gv.input = "channel_signal"
gv.target = "channel_target"

#Organelle to train the model upon
gv.organelle = "Mitochondria" #"Actomyosin-bundles"#"Golgi" #"Plasma-membrane" #"Microtubules" #"Actin-filaments" #"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)" #"Tight-junctions" #"Endoplasmic-reticulum" 

#Assemble the proper tarining csvs by the organelle, model type, and if the data is pertrubed or not
if gv.model_type == "CLF":
    gv.input = "channel_target"
    gv.target = "channel_target"
    gv.train_ds_path = "/home/lionb/cell_generator/image_list_train.csv"
    gv.test_ds_path = "/home/lionb/cell_generator/image_list_test.csv"
else:
    gv.train_ds_path = "/groups/assafza_group/assafza/full_cells_fovs/train_test_list/{}/image_list_train.csv".format(gv.organelle)
    gv.test_ds_path = "/groups/assafza_group/assafza/full_cells_fovs/train_test_list/{}/image_list_test.csv".format(gv.organelle)

#if compound is not None then it will take pertrubed dataset
compound = None #"s-Nitro-Blebbistatin" #"s-Nitro-Blebbistatin" #"Staurosporine" #None #"s-Nitro-Blebbistatin" #None #"paclitaxol_vehicle" #None #"paclitaxol_vehicle" #"rapamycin" #"paclitaxol" #"blebbistatin" #""
#drug could be either the compound or Vehicle which is like DMSO (the unpertrubed data in the pertrubed dataset)
drug = compound #"Vehicle"
if compound is not None:
    ds_path = "/sise/home/lionb/single_cell_training_from_segmentation_pertrub/{}_{}/image_list_test_{}.csv".format(gv.organelle,compound,drug)
else:
    ds_path = gv.train_ds_path
    
gv.batch_size = 4
norm_type = "std"
gv.patch_size = (32,128,128,1)

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
print(gv.organelle)
#example to add predictors to the dataset predictors={"Nuclear-envelope":ne_unet,"Nucleolus-(Granular-Component)":ngc_unet}
train_dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 32, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=0.8,augment=True,norm_type=norm_type, for_clf=for_clf, predictors=None,delete_cahce=True)
validation_dataset = DataGen(ds_path,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 16, patch_size=gv.patch_size,min_precentage=0.8,max_precentage=1.0,augment=False,norm_type=norm_type,for_clf=for_clf,predictors=None)

if (gv.model_type == "VAE"):
    from models.VAE import *
    encoder, layer_dim, filters = get_encoder(gv.patch_size,gv.latent_dim)
    encoder.summary()
    decoder = get_decoder(gv.latent_dim,gv.patch_size,layer_dim,filters)
    decoder.summary()
    model = VAE(encoder,decoder,beta=1) ## beta for reconstruction 1 for KL
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001))

    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        vae_pt = keras.models.load_model(gv.model_path)
        model.set_weights(vae_pt.get_weights())
        
    checkpoint_callback = SaveModelCallback(min(5,gv.number_epochs),model)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20)
    callbacks = [checkpoint_callback,early_stop_callback]
    model.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=callbacks)
    
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
    model = AAE(encoder,decoder,gv.patch_size,discriminator_latent,discriminator_image)
    model.compile(d_latent_optimizer = keras.optimizers.Adam(learning_rate=0.000005), d_image_optimizer = keras.optimizers.Adam(learning_rate=0.000005), g_optimizer = keras.optimizers.Adam(learning_rate=0.00002))

    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        aae_pt = keras.models.load_model(gv.model_path)
        model.set_weights(aae_pt.get_weights())  
    model.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(5,gv.number_epochs),aae)])
    
elif (gv.model_type == "AE"):
    from models.AE import *
    encoder, layer_dim, filters = get_encoder(gv.patch_size)
    encoder.summary()
    decoder = get_decoder(gv.patch_size,layer_dim,filters)
    decoder.summary()
    model = AE(encoder,decoder)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        ae_pt = keras.models.load_model(gv.model_path)
        model.set_weights(ae_pt.get_weights())
        
    model.fit(train_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(5,model)])    

elif (gv.model_type == "UNET"):
    from models.UNETO import *
    num_channels = 1
    model = get_unet((gv.patch_size[0],gv.patch_size[1],gv.patch_size[2],num_channels),activation="linear")
    model.summary()
    
    model = UNET(model,num_channels)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),run_eagerly=True)
    
    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        unet_pt = keras.models.load_model(gv.model_path)                           
        model.set_weights(unet_pt.get_weights())
    checkpoint_callback = SaveModelCallback(min(1,gv.number_epochs),model,gv.model_path)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=150,monitor="val_loss",restore_best_weights=True)
    model.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[checkpoint_callback,early_stop_callback])

elif (gv.model_type == "CLF"):
    from models.Classifier import get_clf
    
    model  = get_clf(gv.patch_size, num_classes=13)
    checkpoint_callback = SaveModelCallback(min(5,gv.number_epochs),model,gv.model_path,monitor="val_loss",save_all=True)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=10,monitor="val_loss",restore_best_weights=True)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),loss=keras.losses.SparseCategoricalCrossentropy(),metrics=[keras.metrics.SparseCategoricalAccuracy()])
    model.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[checkpoint_callback,early_stop_callback])
    
elif (gv.model_type == "UNETP"):
    from models.UNETO_perception import *
    
    model = get_unet(gv.patch_size,activation="relu")
    model.summary()
    
    pl_model,pl_preprocess = get_perception_model(gv.patch_size)
    # model = UNET(unet_model,pl_model,pl_preprocess,gv.patch_size)
    model = get_unet((gv.patch_size[0],gv.patch_size[1],gv.patch_size[2],2),activation="linear")
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),run_eagerly=False)
    
    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        unet_pt = keras.models.load_model(gv.model_path)                           
        model.set_weights(unet_pt.get_weights())  
    checkpoint_callback = SaveModelCallback(min(1,gv.number_epochs),model,gv.model_path,monitor="val_total_loss")
    early_stop_callback = keras.callbacks.EarlyStopping(patience=300,monitor="val_total_loss",restore_best_weights=True)
    model.fit(train_dataset,validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[checkpoint_callback,early_stop_callback])

elif (gv.model_type == "MG"):
    from models.MaskInterpreter import *
    from models.UNETO import *
    similiarity_loss_weight = 0.0  # 1.0 default
    target_loss_weight = 6.0 #10.0 is the default value 
    mask_loss_weight=1.0 #1.0 is the default value 
    noise_scale = 1.5 #value according to find_noise_scale
    
    #The default target score calculation is regular PCC, if one wish to use weighted PCC uncomment the line below
    weighted_pcc = False
    #Uncomment below for modified pearson
    # weighted_pcc = True
    
    
    if weighted_pcc:
        dilate = weighted_pcc #False for regular pearson, True for modified
        gv.target = "structure_seg"
        train_dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 32, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=0.8,augment=True,norm_type=norm_type, for_clf=for_clf, dilate=dilate) 
        validation_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 8, patch_size=gv.patch_size,min_precentage=0.8,max_precentage=1.0,augment=False,norm_type=norm_type,for_clf=for_clf, dilate=dilate)
    
    interpert_model = keras.models.load_model(gv.interpert_model_path)
    interpert_model.summary()
    
    # A model that will create the mask, it's input is a conv layer result with 64 channels of the input and prediction of the interpert model output the importance mask
    adaptor = get_unet((*gv.patch_size[:-1],64),activation="sigmoid") 
    adaptor.summary()
    
    model = MaskInterpreter(gv.patch_size, adaptor, interpert_model, weighted_pcc=weighted_pcc,pcc_target=0.9)
    model.unet.trainable = False
    
    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        mg_pt = keras.models.load_model(gv.model_path)
        model.set_weights(mg_pt.get_weights())
    
    # checkpoint callback monitoring "val_stop" decides when to save the epoch, it is the linear composition of the distance from the target score and the size of the mask self.pcc_target-pcc_loss + mean_mask
    # term and term value make surm that the value of term in the loss is greater then the term value before saving the epoch
    checkpoint_callback = SaveModelCallback(min(1,gv.number_epochs),model,gv.model_path,monitor="val_stop",term="val_pcc",term_value=0.03)
    print("similiarity_loss_weight: ", similiarity_loss_weight)
    print("mask_loss_weight: ",mask_loss_weight)
    print("noise_scale: ",noise_scale)
    print("target_loss_weight:",target_loss_weight)
    
    early_stop_callback = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_total_loss")
    # weight_loss_adaptor_callback = ChangeWeightLossCallbackMaskInterpreter()
    model.compile(g_optimizer = keras.optimizers.Adam(learning_rate=0.0001),mask_loss_weight=mask_loss_weight,noise_scale=noise_scale,target_loss_weight=target_loss_weight,similiarity_loss_weight=similiarity_loss_weight,run_eagerly=True)
    
    # model.save("./model.h5")
    
    losses = model.fit(train_dataset, validation_data=validation_dataset, epochs=100, callbacks=[checkpoint_callback,early_stop_callback])  #weight_loss_adaptor_callback
        
    a = pd.DataFrame(losses.history)
    create_dir_if_not_exist(gv.model_path)
    a.to_csv("{}/losses.csv".format(gv.model_path))
    
elif (gv.model_type == "RC"):
    from models.RegCNN import *
    
    regressor = get_reg(gv.patch_size)
    regressor.summary()
    
    model = RegCNN(regressor)
    
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00001))
    early_stop_callback = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True,monitor="val_loss")
    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        rc_pt = keras.models.load_model(gv.model_path)
        model.set_weights(rc_pt.get_weights())  
    model.fit(train_dataset, validation_data=validation_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(1,gv.number_epochs),model,gv.model_path),early_stop_callback])
    
elif (gv.model_type == "PM"):
    from models.PMCNN import *
    
    model = get_pm((16,64,64,1))
    model.summary()
    
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.000004),loss=keras.losses.binary_crossentropy ,metrics=[keras.metrics.BinaryAccuracy(),keras.metrics.Precision(),keras.metrics.Recall()]) #
    early_stop_callback = keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True,monitor="val_loss")
    if CONTINUE_TRAINING and os.path.exists(gv.model_path):
        pm_pt = keras.models.load_model(gv.model_path)
        model.set_weights(pm_pt.get_weights())  
    FINE_TUNE = False
    if FINE_TUNE:
        for layer in model.layers:
            if layer.name != "pair_matching_denseout" and layer.name != "pair_matching_dense2":
                layer.trainable = False
        
    train_dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 32, patch_size=gv.patch_size,min_precentage=0,max_precentage=0.9,augment=True,pairs=True,neg_ratio=1,cutoff=0.02,masking_pair=True)
    validation_dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 8, patch_size=gv.patch_size,min_precentage=0.9,max_precentage=1,augment=False,pairs=True,neg_ratio=1,cutoff=0.02,masking_pair=True)
    model.fit(train_dataset, validation_data=validation_dataset, epochs=1000 ,callbacks=[SaveModelCallback(min(3,gv.number_epochs),model,gv.model_path),early_stop_callback])

model.save(gv.model_path,save_format="tf")