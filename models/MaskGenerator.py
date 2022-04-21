import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv


def get_adaptor(input_size,layers=[32,32,32,32],name="adaptor"):
        
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
        
        output = conv_layer(filters=1,kernel_size=16,strides=1,padding='same',activation='relu',kernel_regularizer="l1", dtype=tf.float32,name="{}_conv_out".format(name))(x)
        
        return keras.Model(input,output,name=name)

   
class MaskGenerator(keras.Model):
    def __init__(self, patch_size, adaptor, unet, **kwargs):
        super(MaskGenerator, self).__init__(**kwargs)

        self.unet = unet
        
        image_input = keras.layers.Input(shape=patch_size)
        processed_image = keras.layers.Conv3D(filters=32,kernel_size=3,padding="same",activation="relu")(image_input)
        # processed_image = image_input
        mask = adaptor(processed_image)
        self.generator = keras.Model(image_input,mask,name="generator")
        
        self.unet_loss_tracker = keras.metrics.Mean(
            name="unet_loss"
        )
        
        self.mask_loss_tracker = keras.metrics.Mean(
            name="mask_loss"
        )
        
        self.mask_acc = keras.metrics.BinaryAccuracy(
            name="mask_acc"
        )
        
        self.total_loss_tracker = keras.metrics.Mean(name = "total")
        

    
    def compile(self, g_optimizer, unet_loss_weight=1, mask_loss_weight=0.00):
        super(MaskGenerator, self).compile()
        self.g_optimizer = g_optimizer
        self.unet_loss_weight = unet_loss_weight
        self.mask_loss_weight = mask_loss_weight
    
    @property
    def metrics(self):
        return [
            self.unet_loss_tracker,
            self.mask_loss_tracker,
            self.total_loss_tracker,
            self.mask_acc
        ]

    def train_step(self, data, train=True):
        data_0 = data[0] #tf.cast(data[0],dtype=tf.float16)
        # data_1 = tf.cast(data[1],dtype=tf.float16)
        
        unet_target = self.unet(data_0)
        
        # Train generator to fool discriminator_image and create target
        with tf.GradientTape() as tape:
            mask = self.generator(data_0)
            # avg = tf.reduce_mean(data_0)
            # adapted_image = tf.where(mask>0.5,data_0,avg)
            adapted_image = mask*data_0
            unet_predictions = self.unet(adapted_image)
            
            unet_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(unet_target, unet_predictions),axis=(1,2)
                ),axis=(0,1)
            )
            
            mask_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_absolute_error(tf.zeros_like(data_0), mask),axis=(1,2)
                ),axis=(0,1)
            )
            self.mask_loss_tracker.update_state(mask_loss)
            self.mask_acc.update_state(tf.zeros_like(data_0), mask)
            self.unet_loss_tracker.update_state(unet_loss)  
             
            total_loss = (mask_loss*self.mask_loss_weight) + unet_loss*self.unet_loss_weight
            # total_loss = total_loss / (self.discriminator_loss_weight + self.unet_loss_weight + self.mask_loss_weight)
            self.total_loss_tracker.update_state(total_loss)
            
        if (train):
            grads = tape.gradient(total_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights)) 
        
               
        return {
            "unet": self.unet_loss_tracker.result(),
            "mask": self.mask_loss_tracker.result(),
            "total loss":self.total_loss_tracker.result(),
            "mask_acc": self.mask_acc.result()
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        return self.generator(input)