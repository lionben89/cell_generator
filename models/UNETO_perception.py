import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv
from tensorflow.keras.applications import vgg19

def get_unet(input_size,activation="relu",regularizer=None,name="unet"):
        
        if (input_size[0]==1): ## 2D image
            input_size = input_size[1:]
            
        if len(input_size) == 4:
            conv_layer = keras.layers.Conv3D
            convt_layer = keras.layers.Conv3DTranspose
        else:
            conv_layer = keras.layers.Conv2D
            convt_layer = keras.layers.Conv2DTranspose
        
        layer_dim = np.array(input_size[:-1],dtype=np.int32)
        filters = 16
        
        input = keras.Input(shape=input_size,dtype=tf.float16)
        skip_connection = []
        x = input
        ## downsampling layers          
        while layer_dim[0] > 4:
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            # i_x = x
            x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_conv1_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn1_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.ReLU()(x)
            x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_conv2_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn2_{}".format(name,layer_dim[0]))(x)
            x =keras.layers.ReLU()(x)
            skip_connection.append(x)
            # x = i_x + x
            x = conv_layer(filters=filters,kernel_size=4,strides=2,padding='same',name="{}_convd_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bnd_{}".format(name,layer_dim[0]))(x)            
            x =keras.layers.ReLU()(x)
            
            
        ## bottleneck layer
        # filters=filters*2
        # i_x = x
        x = conv_layer(filters=filters*2,kernel_size=3,strides=1,padding='same',name="{}_conv_bottleneck1".format(name))(x)   
        if (gv.batch_norm):
            x = keras.layers.BatchNormalization(name="{}_bn_bottleneck1".format(name))(x)     
        x =keras.layers.ReLU()(x)
        x = convt_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_convt_bottleneck2".format(name))(x)
        if (gv.batch_norm):
            x = keras.layers.BatchNormalization(name="{}_bn_bottleneck2".format(name))(x)        
        x =keras.layers.ReLU()(x)
        # x = i_x + x
        
        ## upsampling layers 
        i = len(skip_connection)-1
        while layer_dim[0] < input_size[0]:
            x = convt_layer(filters=filters,kernel_size=4,strides=2,padding='same',name="{}_convtu_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bntu_{}".format(name,layer_dim[0]))(x)  
            x =keras.layers.ReLU()(x)   
            x = tf.concat([x,skip_connection[i]],axis=-1)        
            # i_x = x
            layer_dim = np.int32(layer_dim * 2)
            x = convt_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_convt1_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bnt1_{}".format(name,layer_dim[0]))(x)  
            x =keras.layers.ReLU()(x)
            x = convt_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_convt2_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bnt2_{}".format(name,layer_dim[0]))(x)                  
            x =keras.layers.ReLU()(x)
            filters = filters / 2
            # x = i_x + x
            i=i-1              
      

        ## last conv same resulotion
        output = conv_layer(filters=1, kernel_size=3,strides=1,padding='same',activity_regularizer=regularizer,activation=activation,dtype=tf.float32,name="{}_conv_{}_out".format(name,layer_dim[0]))(x)
#kernel_regularizer=regularizer
        return keras.Model(input,output,name=name)  

def get_perception_model(input_size):
    pl_model = vgg19.VGG19(input_shape=(input_size[1],input_size[2],3),include_top=False,weights='imagenet')
    pl_model.trainable=False
    outputs=[]
    for layer in pl_model.layers:
        if layer.name.endswith('conv2'):
            outputs.append(layer.output)
    pl_preprocess = vgg19.preprocess_input
    new_pl_model = keras.Model(inputs=pl_model.input, outputs=outputs)
    return new_pl_model,pl_preprocess
    
class UNET(keras.Model):
    def __init__(self, unet, pl_model, pl_preprocess, input_size, **kwargs):
        super(UNET, self).__init__(**kwargs)
        
        self.input_size = input_size
        
        self.unet = unet
        
        self.pl_model = pl_model
        self.pl_preprocess = pl_preprocess
        
        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )
        self.pl_loss_tracker = keras.metrics.Mean(
            name="pl_loss"
        )
        
        self.total_loss_tracker = keras.metrics.Mean(
            name="total_loss"
        )
        # self.acc_tracker = keras.metrics.MeanIoU(2,name="iou")
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            # self.acc_tracker
            self.pl_loss_tracker,
            self.total_loss_tracker
        ]

    def train_step(self, data, train=True):
        # data = tf.cast(data,dtype=tf.float16)
        batch_size = tf.shape(data[0])[0]
        # num_z = tf.shape(data[0])[1]
        
        # Train unet
        with tf.GradientTape() as tape:
            prediction = self.unet(data[0])
           
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data[1], prediction),axis=(1,2,3)
                ),axis=0
            )
            
            pl_loss = tf.constant(0.0,dtype=tf.float32)
            for k in range(self.input_size[0]): ## for z slice
                original_perception = self.pl_model(self.pl_preprocess(tf.repeat(data[1][:,k],repeats=3,axis=-1)*255.))
                prediction_perception = self.pl_model(self.pl_preprocess(tf.repeat(prediction[:,k],repeats=3,axis=-1)*255.))
                for i in range(len(original_perception)):
                    pl_loss = pl_loss + tf.reduce_mean(keras.losses.mean_squared_error(tf.cast(original_perception[i],dtype=tf.float32)*0.01,tf.cast(prediction_perception[i],dtype=tf.float32)*0.01))
                    
            self.pl_loss_tracker.update_state(pl_loss)
            # loss = keras.losses.binary_crossentropy(data[1], prediction)
            total_loss = pl_loss+0.01*loss
            self.total_loss_tracker.update_state(total_loss)

        if (train):
            grads = tape.gradient(total_loss, self.unet.trainable_weights) #+0.01*loss
            self.optimizer.apply_gradients(zip(grads, self.unet.trainable_weights))
        self.loss_tracker.update_state(loss)
        # self.acc_tracker.update_state(data[1], tf.math.round(prediction))
        
        
        return {
            "loss": self.loss_tracker.result(),
            "pl_loss": self.pl_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result()
            # "iou": self.acc_tracker.result()
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        prediction = (self.unet(input))
        return prediction