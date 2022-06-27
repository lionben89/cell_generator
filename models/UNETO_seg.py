import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv


def get_unet(input_size,activation="relu",name="unet"):
        
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
        while layer_dim[0] > 2:
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            # i_x = x
            x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_conv1_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn1_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.ReLU()(x)
            # x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_conv2_{}".format(name,layer_dim[0]))(x)
            # if (gv.batch_norm):
            #     x = keras.layers.BatchNormalization(name="{}_bn2_{}".format(name,layer_dim[0]))(x)
            # x =keras.layers.ReLU()(x)
            skip_connection.append(x)
            # x = i_x + x
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',name="{}_convd_{}".format(name,layer_dim[0]))(x)
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
            x = convt_layer(filters=filters,kernel_size=3,strides=2,padding='same',name="{}_convtu_{}".format(name,layer_dim[0]))(x)
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
            # x = convt_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_convt2_{}".format(name,layer_dim[0]))(x)
            # if (gv.batch_norm):
            #     x = keras.layers.BatchNormalization(name="{}_bnt2_{}".format(name,layer_dim[0]))(x)                  
            # x =keras.layers.ReLU()(x)
            filters = filters / 2
            # x = i_x + x
            i=i-1              
      

        ## last conv same resulotion
        output = conv_layer(filters=1, kernel_size=3,strides=1,padding='same',activation=activation,dtype=tf.float32,name="{}_conv_{}_out".format(name,layer_dim[0]))(x)

        return keras.Model(input,output,name=name)  

class UNET(keras.Model):
    def __init__(self, unet,pre_unet, **kwargs):
        super(UNET, self).__init__(**kwargs)
        
        self.unet = unet
        self.pre_unet = pre_unet
        
        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )
        self.acc_tracker = keras.metrics.MeanIoU(2,name="iou")
        
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.acc_tracker
        ]

    def train_step(self, data, train=True):
        # data = tf.cast(data,dtype=tf.float16)
        batch_size = tf.shape(data[0])[0]
        
        # Train unet
        with tf.GradientTape() as tape:
            prediction = self.unet(self.pre_unet(data[0]))
           
            loss = 0.001*tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data[1], prediction),axis=(1,2,3)
                ),axis=0
            )
            # loss = keras.losses.binary_crossentropy(data[1], prediction)
            

        if (train):
            grads = tape.gradient(loss, self.unet.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.unet.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(data[1], tf.math.round(prediction))
        
        
        return {
            "loss": self.loss_tracker.result(),
            "iou": self.acc_tracker.result()
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        prediction = self.unet(self.pre_unet(input))
        return prediction