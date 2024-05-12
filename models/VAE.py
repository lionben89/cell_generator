import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

"""Variational Auto Encoder"""
class Sampling(keras.layers.Layer):
    """Uses (mu, sigma) to sample z."""

    def call(self, inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(sigma)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim),dtype=tf.float16)
        return mu + sigma * epsilon

def get_encoder(input_size,latent_dim,name="encoder"):
        
        if (input_size[0]==1): ## 2D image
            input_size = input_size[1:]
            
        if len(input_size) == 4:
            conv_layer = keras.layers.Conv3D
        else:
            conv_layer = keras.layers.Conv2D
        
        layer_dim = np.array(input_size[:-1],dtype=np.int32)
        filters = 8
        
        input = keras.Input(shape=input_size)
        
        ## first conv same resulotion
        x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(input)
        x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
        
        
        ## downsampling layers          
        while layer_dim[0] > 1:
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
            

        ## flatten + dense layer
        x = keras.layers.BatchNormalization(name="{}_bn_dense_{}".format(name,layer_dim[0]))(x)
        x = keras.layers.Flatten()(x) #keras.layers.Dense(tf.math.reduce_prod(layer_dim), activation='relu', name="{}_fc_1".format(name))(keras.layers.Flatten()(x))
        
        
        ## mu
        mu = keras.layers.Dense(latent_dim, name="{}_mu".format(name))(x)
        
        ## sigma
        sigma = keras.layers.Dense(latent_dim, name="{}_sigma".format(name))(x)
        
        ## z
        z = Sampling(name="{}_z".format(name))([mu,sigma])
        
        return keras.Model(input,[mu,sigma,z],name=name), np.int32((*layer_dim,filters)), filters

        
def get_decoder(latent_dim,output_size,layer_dim,filters,name="decoder"):
        
        if (output_size[0]==1): ## 2D image
            output_size = output_size[1:]
        
        if len(output_size) == 4:
            conv_layer = keras.layers.Conv3DTranspose
        else:
            conv_layer = keras.layers.Conv2DTranspose
        
        input = keras.Input(shape=(latent_dim,))
        
        ## unflatten + dense layer
        x = keras.layers.Reshape(layer_dim)(keras.layers.Dense(tf.math.reduce_prod(layer_dim),name="{}_fc_1".format(name))(input))
        x = keras.layers.BatchNormalization(name="{}_bn_dense_{}".format(name,layer_dim[0]))(x)

        x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
        x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
        
        ## upsampling layers  
        while layer_dim[0] < output_size[0]:
            layer_dim = layer_dim * 2
            filters = filters // 2
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
      

        ## last conv same resulotion
        output = conv_layer(filters=1, kernel_size=3,strides=1,padding='same',kernel_initializer='glorot_normal',activation="relu",name="{}_conv_{}_out".format(name,layer_dim[0]))(x)

        return keras.Model(input,output,name=name)  
    
class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=0.5, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.beta = beta
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data, train=True):
        data = tf.cast(data,dtype=tf.float16)
        with tf.GradientTape() as tape:
            mu, sigma, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data[1], reconstruction),axis=(2,3)
                )
            )
            batch = tf.shape(mu)[0]
            dim = tf.shape(sigma)[1]
            kl_loss = tf.reduce_mean(keras.losses.kl_divergence(keras.backend.random_normal(shape=(batch, dim),dtype=tf.float16),z))
            total_loss = self.beta*reconstruction_loss + kl_loss
        if train:            
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }      
    
    def test_step(self, data):    
        return self.train_step(data,False)
    
    def call(self,input):
        mu, sigma, z = self.encoder(input)
        reconstruction = self.decoder(z)
        return reconstruction