import tensorflow as tf
import tensorflow.keras as keras
from dataset import PatchDataGen
from models.VAE import *
import matplotlib.pyplot as plt
tf.compat.v1.enable_eager_execution()

CONTINUE_TRAINING = False

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
model_path = "./vae_single_cell_2d"
patch_size = (32,64,64,1)
latent_dim = 64
train_dataset = PatchDataGen("/sise/home/lionb/cell_generator/image_list_train.csv","channel_signal","channel_signal",128,patch_size=patch_size)
validation_dataset = PatchDataGen("/sise/home/lionb/cell_generator/image_list_test.csv","channel_signal","channel_signal",8,patch_size=patch_size)

encoder, layer_dim, filters = get_encoder(patch_size,latent_dim)
encoder.summary()
decoder = get_decoder(latent_dim,patch_size,layer_dim,filters)
decoder.summary()
vae = VAE(encoder,decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

if CONTINUE_TRAINING:
    vae_pt = keras.models.load_model(model_path)
    vae.set_weights(vae_pt.get_weights())
    
# callback = keras.callbacks.ModelCheckpoint("./vae_single_cell.h5",monitor="val_loss", save_best_only=True)
vae.fit(train_dataset,validation_data=validation_dataset, epochs=700)#, callbacks=callback) ## validation_data=validation_dataset,
vae.save(model_path,save_format="tf")