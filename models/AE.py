import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

def get_encoder(input_size,name="encoder"):
        
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
        while layer_dim[0] > 4:
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
            

        ## flatten + dense layer
        z = keras.layers.BatchNormalization(name="{}_bn_dense_{}".format(name,layer_dim[0]))(x)
        

        return keras.Model(input,z,name=name), np.int32((*layer_dim,filters)), filters

        
def get_decoder(output_size,layer_dim,filters,name="decoder"):
        
        if (output_size[0]==1): ## 2D image
            output_size = output_size[1:]
        
        if len(output_size) == 4:
            conv_layer = keras.layers.Conv3DTranspose
        else:
            conv_layer = keras.layers.Conv2DTranspose
        
        input = keras.Input(shape=(layer_dim))
        
        x = keras.layers.BatchNormalization(name="{}_bn_dense_{}".format(name,layer_dim[0]))(input)

        x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
        x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
        
        ## upsampling layers  
        while layer_dim[0] < output_size[0]:
            layer_dim = layer_dim * 2
            filters = filters // 2
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
      

        ## last conv same resulotion
        output = conv_layer(filters=1, kernel_size=3,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}_out".format(name,layer_dim[0]))(x)

        return keras.Model(input,output,name=name)  
    
class AE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker
        ]

    def train_step(self, data):

        with tf.GradientTape() as tape:
            z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data[1], reconstruction),axis=(1,2)
                )
            )
            
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }      
    
    def test_step(self, data):    
        z = self.encoder(data[0])
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(data[1], reconstruction),axis=(1,2)
            )
        )

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        } 
    
    def call(self,input):
        z = self.encoder(input)
        reconstruction = self.decoder(z)
        return reconstruction