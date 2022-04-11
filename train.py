import os
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
import global_vars as gv
from callbacks import *
from metrics import *

tf.compat.v1.enable_eager_execution()
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# CONTINUE_TRAINING = False
CONTINUE_TRAINING = True

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

train_dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 108, patch_size=gv.patch_size,min_precentage=0,max_precentage=1)
validation_dataset = DataGen(gv.test_ds_path,gv.input,gv.target,batch_size = 4, num_batches = 4, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1,augment=False)


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
    
    unet_model = get_unet(gv.patch_size,activation="sigmoid")
    unet_model.summary()
    
    unet_model = UNET(unet_model)
    # unet_model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),loss=keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=0.1)) #,metrics=keras.metrics.MeanIoU(2)
    unet_model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0002))
    
    if CONTINUE_TRAINING and os.path.exists(gv.unet_model_path):
        unet_pt = keras.models.load_model(gv.unet_model_path)
        unet_model.set_weights(unet_pt.get_weights())  
    checkpoint_callback = SaveModelCallback(min(5,gv.number_epochs),unet_model,gv.unet_model_path)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=25)
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
    
elif (gv.model_type == "EAM"): #Explainable Adverserial Model
    from models.EAM import *
    from models.UNETO import *
    
    unet = keras.models.load_model(gv.unet_model_path)
    unet.summary()
    
    # adaptor = get_adaptor(gv.patch_size)
    # adaptor.summary()
    adaptor = get_unet(gv.patch_size,activation="relu")
    adaptor.summary()
    
    eam = EAM(gv.patch_size,adaptor, unet)
    
    eam.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00001))
    if CONTINUE_TRAINING and os.path.exists(gv.sg_model_path):
        eam_pt = keras.models.load_model(gv.aem_model_path)
        eam.set_weights(eam_pt.get_weights())  
    eam.fit(train_dataset, epochs=gv.number_epochs, callbacks=[SaveModelCallback(min(5,gv.number_epochs),eam,gv.eam_model_path)])
    eam.save(gv.eam_model_path,save_format="tf")

elif (gv.model_type=="VNET"):
    from keras_unet.models import custom_vnet
    model = custom_vnet(
        input_shape=gv.patch_size,
        num_classes=1,
        activation="relu",
        use_batch_norm=True,
        upsample_mode="deconv",  # 'deconv' or 'simple'
        dropout=0.3,
        dropout_change_per_layer=0.0,
        dropout_type="spatial",
        use_dropout_on_upsampling=True,
        use_attention=True,
        filters=32,
        num_layers=4,
        output_activation="relu",
    ) # 'sigmoid' or 'softmax'

    """
    Customizable VNet architecture based on the work of
    Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi in
    V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

    Arguments:
    input_shape: 4D Tensor of shape (x, y, z, num_channels)

    num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation

    activation (str): A keras.activations.Activation to use. ReLu by default.

    use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers

    upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part

    dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off

    dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block

    dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]

    use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network

    use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]

    filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block

    num_layers (int): Number of total layers in the encoder not including the bottleneck layer

    output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation

    Returns:
    model (keras.models.Model): The built V-Net

    Raises:
    ValueError: If dropout_type is not one of "spatial" or "standard"


    [1]: https://arxiv.org/abs/1505.04597
    [2]: https://arxiv.org/pdf/1411.4280.pdf
    [3]: https://arxiv.org/abs/1804.03999

    """
    from keras_unet.metrics import iou, iou_thresholded
    from keras_unet.losses import jaccard_distance
    


    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(), 
        #optimizer=SGD(lr=0.01, momentum=0.99),
        # loss='binary_crossentropy',
        loss='mse',
        #metrics=[iou, iou_thresholded]
    )
    history = model.fit_generator(
        train_dataset,
        steps_per_epoch=64,
        epochs=gv.number_epochs,
        validation_data=validation_dataset,
        callbacks=[SaveModelCallback(min(5,gv.number_epochs),model,gv.unet_model_path)]
    )
    from keras_unet.utils import plot_segm_history

    plot_segm_history(history)