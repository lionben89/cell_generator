import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import tensorflow_addons as tfa

from metrics import tf_pearson_corr
tf.compat.v1.enable_eager_execution()

# class BlobRegularizer(keras.regularizers.Regularizer):
    
#     def __init__(self, l1=0.0,l2=0.0,blob=0.0,kernel=3):
#         self.l1 = l1
#         self.l2 = l2
#         self.blob = blob
#         self.kernel = kernel

#     def __call__(self, x):
#         x = tf.cast(x,dtype=tf.float32)
#         x = tf.round(x)
#         blob_mask = -tf.nn.max_pool3d(-x, ksize=[self.kernel,self.kernel,1], strides=1,padding="SAME", name='erosion3D')
#         blob_mask = tf.nn.max_pool3d(blob_mask, ksize=[self.kernel,self.kernel,1], strides=1, padding="SAME", name='dilation3D')
#         blob_loss = tf.reduce_sum(tf.maximum((x-blob_mask),tf.zeros_like(x)))
#         # blob_loss = tf.reduce_sum(tf.abs((x-blob_mask)))
#         l1 = tf.reduce_sum(tf.abs(x))
#         l2 = tf.reduce_sum(tf.square(x))
#         return self.blob * blob_loss + self.l1*l1 + self.l2*l2
    
    # def get_config(self):
    #     return {'l1': self.l1,'kernel':self.kernel,'blob':self.blob,'l2':self.l2}

class MaskGenerator(keras.Model):
    def __init__(self, patch_size, adaptor, unet, weighted_pcc,pcc_target=0.9, **kwargs):
        super(MaskGenerator, self).__init__(**kwargs)

        self.weighted_pcc = weighted_pcc
        self.pcc_target = pcc_target
        self.unet = unet
        
        image_input = keras.layers.Input(shape=patch_size,dtype=tf.float16)
        target_input = keras.layers.Input(shape=patch_size,dtype=tf.float16)
        processed_target = keras.layers.Conv3D(filters=32,kernel_size=3,padding="same",activation="relu")(target_input)
        processed_image = keras.layers.Conv3D(filters=32,kernel_size=3,padding="same",activation="relu")(image_input)
        processed_input = keras.layers.Concatenate(axis=-1)([processed_image,processed_target])
        # processed_input = processed_image
        mask = tf.cast(adaptor(processed_input),dtype=tf.float32)
        # mask = -tf.nn.max_pool3d(-mask, ksize=3, strides=1,padding="SAME", name='erosion3D')
        # mask = tf.nn.max_pool3d(mask, ksize=3, strides=1, padding="SAME", name='dilation3D')
        mask = tf.cast(mask,dtype=tf.float64)
        self.generator = keras.Model([image_input,target_input],mask,name="generator")
        # self.generator = keras.Model(image_input,mask,name="generator")
        
        self.unet_loss_tracker = keras.metrics.Mean(
            name="unet_loss"
        )
        
        self.blob_ratio = keras.metrics.BinaryAccuracy(
            name="blob_ratio"
        )
        
        self.mask_ratio = keras.metrics.Mean(
            name="mask_ratio"
        )
        
        self.mask_size = keras.metrics.Mean(
            name="mask_size"
        )
        
        self.total_loss_tracker = keras.metrics.Mean(name = "total")
        
        self.pcc = keras.metrics.Mean(name = "pcc")
        
        self.stop = keras.metrics.Mean(name = "stop")
        

    
    def compile(self, g_optimizer, unet_loss_weight=1. ,mask_loss_weight=0.01,mask_size_loss_weight=0.002,run_eagerly=False, noise_scale=5.0):
        super(MaskGenerator, self).compile(run_eagerly=run_eagerly)
        self.g_optimizer = g_optimizer
        self.unet_loss_weight = unet_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.mask_size_loss_weight = mask_size_loss_weight
        self.noise_scale = noise_scale
    
    @property
    def metrics(self):
        return [
            self.unet_loss_tracker,
            self.total_loss_tracker,
            self.blob_ratio,
            self.mask_ratio,
            self.pcc,
            self.stop,
            self.mask_size
        ]

    def train_step(self, data, train=True):
        data_0 = tf.cast(data[0],dtype=tf.float64)
        data_1 = tf.cast(data[1],dtype=tf.float64)
        
        unet_target = self.unet(data_0)
        
        # Train generator to fool discriminator_image and create target
        
        with tf.GradientTape() as tape:
            mask = self.generator([data_0,unet_target])
            # mask = tf.cast(tf.where(mask>0.5,1.0,mask),dtype=tf.float64)
            # mask = self.generator(data_0)
                    
            # mask_rounded_NOT_differentiable = tf.cast(tf.where(mask>0.5,1.0,0.0),dtype=tf.float64)#tf.round(mask)
            # mask = mask - tf.stop_gradient(mask - mask_rounded_NOT_differentiable)
            
            
            # # Creating kernel
            # kernel = 3
            
            # # Using opening method , ToDo: check if works on 3d
            # blob_mask = -tf.nn.max_pool3d(-tf.cast(mask,dtype=tf.float32), ksize=kernel, strides=1,padding="SAME", name='erosion3D')
            # blob_mask = tf.nn.max_pool3d(blob_mask, ksize=kernel, strides=1, padding="SAME", name='dilation3D')
            # blob_mask = tf.cast(blob_mask,dtype=tf.float64)
            # blur_image = tfa.image.gaussian_filter2d(data_0[:,:,:,:,0],[3,3],3,"CONSTANT",0)
            # blur_image = tf.expand_dims(blur_image,axis=-1)
            
            # adapted_image = tf.where(mask>0,data_0,blur_image)
            # adapted_image = mask*data_0#+(1-blob_mask)*blur_image
            
            normal_noise = tf.random.normal(tf.shape(mask),stddev=self.noise_scale,dtype=tf.float64) 
            # normal_noise = tf.random.uniform(tf.shape(mask),maxval=self.noise_scale,dtype=tf.float64) 
            adapted_image = (mask*data_0)+(normal_noise*(1-mask))
            # inv_adapted_image = (normal_noise*data_0)+(data_0*(1-mask))
            
            unet_predictions = self.unet(adapted_image)
            # inv_unet_predictions = self.unet(inv_adapted_image)
            
            # adapted_image_binary = (mask_binary*data_0)+(normal_noise*(1-mask_binary))
            
            # unet_predictions_binary = self.unet(adapted_image_binary)
            
            # unet_loss = keras.losses.mean_squared_error(unet_target, unet_predictions)  
            unet_loss = tf.reduce_mean(
               tf.reduce_sum(
                   keras.losses.mean_squared_error(unet_target, unet_predictions),axis=(1,2)
               ),axis=(0,1)
            )

            # blob_loss = tf.reduce_sum(tf.maximum((mask-blob_mask),tf.zeros_like(mask)))
            # # blob_loss = tf.reduce_sum(tf.abs((x-blob_mask)))
            # l1 = tf.reduce_mean(tf.abs(blob_mask))
            # l2 = tf.reduce_mean(tf.square(blob_mask))                
            
            mean_mask = tf.reduce_mean((mask)) #tf.math.log
            # mask_size = tf.reduce_mean(mask_binary)
            
            # mask_size_loss = tf.reduce_mean(
            #    tf.reduce_sum(
            #       keras.losses.mean_squared_error(tf.zeros_like(mask), mask_binary),axis=(1,2)
            #    ),axis=(0,1)
            # )
            mask_loss = tf.reduce_mean(
               tf.reduce_sum(
                  keras.losses.mean_squared_error(tf.zeros_like(mask), mask),axis=(1,2)
               ),axis=(0,1)
            )
            
            # blob_mask_loss = tf.reduce_mean(
            #    tf.reduce_sum(
            #       keras.losses.mean_squared_error(tf.zeros_like(blob_mask), blob_mask),axis=(1,2)
            #    ),axis=(0,1)
            # )
            
            if self.weighted_pcc:
                pcc_loss = tf.clip_by_value((tf_pearson_corr(unet_target,unet_predictions,data_1)),-1.0,self.pcc_target)
            else:
                pcc_loss = tf.clip_by_value((tf_pearson_corr(unet_target,unet_predictions)),-1.0,self.pcc_target)
            # inv_pcc_loss = tf.math.abs(tf_pearson_corr(unet_target,inv_unet_predictions))
            
            total_loss = 0.1*unet_loss + (mask_loss)*self.mask_loss_weight + (self.pcc_target-pcc_loss)*5000 #+ inv_pcc_loss*1000 #+ mask_size_loss*self.mask_size_loss_weight
            
        if (train):
            grads = tape.gradient(total_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights)) 
        
        self.mask_ratio.update_state((1-mask)) #tf.reduce_sum(mask)
        self.unet_loss_tracker.update_state(unet_loss)  
        self.blob_ratio.update_state(tf.zeros_like(mask), mask)
        self.pcc.update_state(pcc_loss)
        self.total_loss_tracker.update_state(total_loss)
        # self.mask_size.update_state(inv_pcc_loss)
        self.stop.update_state(self.pcc_target-pcc_loss + mean_mask)
        
               
        return {
            "unet": self.unet_loss_tracker.result(),
            "binary_ratio": self.blob_ratio.result(),
            "mask_ratio": self.mask_ratio.result(),
            "total loss":self.total_loss_tracker.result(),
            # "inv_pcc": self.mask_size.result(),
            "pcc":self.pcc.result(),
            "stop": self.stop.result()
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        original_unet = self.unet(input)
        return self.generator([input,original_unet])
        # return self.generator(input)