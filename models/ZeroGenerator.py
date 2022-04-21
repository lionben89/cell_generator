import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv


def get_adaptor(input_size,layers=[64,64,64,64],name="adaptor"):
        
        if (input_size[0]==1): ## 2D image
            input_size = input_size[1:]
            
        if len(input_size) == 4:
            conv_layer = keras.layers.Conv3D
        else:
            conv_layer = keras.layers.Conv2D
        
        input = keras.Input(shape=input_size)

        x = input
        
        ## adaptor layers          
        for i in range(len(layers)):
            x = conv_layer(filters=layers[i],kernel_size=3,strides=1,padding='same',activation='relu',name="{}_conv_{}".format(name,i))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,i))(x)
        
        output = conv_layer(filters=1,kernel_size=3,strides=1,padding='same',activation='relu',dtype=tf.float32,name="{}_conv_out".format(name))(x)
        
        return keras.Model(input,output,name=name)

   
class ZeroGenerator(keras.Model):
    def __init__(self, patch_size, adaptor, unet, **kwargs):
        super(ZeroGenerator, self).__init__(**kwargs)

        self.unet = unet
        
        image_input = keras.layers.Input(shape=patch_size)
        processed_image = keras.layers.Conv3D(filters=32,kernel_size=3,padding="same",activation="relu")(image_input)
        # processed_image = image_input
        output = adaptor(processed_image)
        self.generator = keras.Model(image_input,output,name="generator")
        
        self.unet_loss_tracker = keras.metrics.Mean(
            name="unet_loss"
        )
        
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        
        self.total_loss_tracker = keras.metrics.Mean(name = "total")
        

    
    def compile(self, g_optimizer, unet_loss_weight=1, reconstruction_loss_weight=0.0):
        super(ZeroGenerator, self).compile()
        self.g_optimizer = g_optimizer
        self.unet_loss_weight = unet_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
    
    @property
    def metrics(self):
        return [
            self.unet_loss_tracker,
            self.reconstruction_loss_tracker,
            self.total_loss_tracker
        ]

    def train_step(self, data, train=True):
        data_0 = data[0] #tf.cast(data[0],dtype=tf.float16)
        # data_1 = tf.cast(data[1],dtype=tf.float16)
        
        unet_target = self.unet(data_0)
        
        # Train generator to fool discriminator_image and create target
        with tf.GradientTape() as tape:
            adapted_image = self.generator(data_0)
            
            unet_predictions = self.unet(adapted_image)
            
            unet_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(unet_target, unet_predictions),axis=(1,2)
                ),axis=(0,1)
            )
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_absolute_error(tf.zeros_like(data_0), adapted_image),axis=(1,2,3)
                ),axis=0
            )
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.unet_loss_tracker.update_state(unet_loss)  
             
            total_loss = (reconstruction_loss*self.reconstruction_loss_weight) + unet_loss*self.unet_loss_weight
            # total_loss = total_loss / (self.discriminator_loss_weight + self.unet_loss_weight + self.reconstruction_loss_weight)
            self.total_loss_tracker.update_state(total_loss)
            
        if (train):
            grads = tape.gradient(total_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights)) 
        
               
        return {
            "unet": self.unet_loss_tracker.result(),
            "reconstruction": self.reconstruction_loss_tracker.result(),
            "total loss":self.total_loss_tracker.result()
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        return self.generator(input)