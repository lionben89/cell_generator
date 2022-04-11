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
        while layer_dim[0] > 1:
            # i_x = x
            x = conv_layer(filters=filters,kernel_size=7,strides=1,padding='same',name="{}_conv1_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.Dense(4*filters)(x)
            x = keras.layers.Activation("gelu")(x)
            x = keras.layers.Dense(filters)(x)

            # x = i_x+x
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',name="{}_convd2_{}".format(name,layer_dim[0]))(x)
            skip_connection.append(x)
            
        ## bottleneck layer
        x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',activation='relu',name="{}_conv_bottleneck1".format(name))(x)
        x = keras.layers.Dense(4*filters)(x)
        x = keras.layers.Activation("gelu")(x)
        x = keras.layers.Dense(filters)(x)  
        x = convt_layer(filters=filters,kernel_size=3,strides=1,padding='same',activation='relu',name="{}_convt_bottleneck2".format(name))(x)
        x = keras.layers.Dense(4*filters)(x)
        x = keras.layers.Activation("gelu")(x)
        x = keras.layers.Dense(filters)(x)     
        filters=filters*2    
        ## upsampling layers 
        i = len(skip_connection)-1
        while layer_dim[0] < input_size[0]:
            x = tf.concat([x,skip_connection[i]],axis=-1)
            # i_x = x
            x = convt_layer(filters=filters,kernel_size=7,strides=1,padding='same',name="{}_convt1_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.Dense(4*filters)(x)
            x = keras.layers.Activation("gelu")(x)
            x = keras.layers.Dense(filters)(x)                  
            # x = i_x+x               
            layer_dim = layer_dim * 2
            filters = filters // 2
            x = convt_layer(filters=filters,kernel_size=3,strides=2,padding='same',name="{}_convtu2_{}".format(name,layer_dim[0]))(x)
                
            i=i-1              
      

        ## last conv same resulotion
        output = conv_layer(filters=1, kernel_size=3,strides=1,padding='same',activation=activation,dtype=tf.float32,name="{}_conv_{}_out".format(name,layer_dim[0]))(x)

        return keras.Model(input,output,name=name)  

class UNET(keras.Model):
    def __init__(self, unet, **kwargs):
        super(UNET, self).__init__(**kwargs)
        
        self.unet = unet
        
        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    def train_step(self, data, train=True):
        # data = tf.cast(data,dtype=tf.float16)
        batch_size = tf.shape(data[0])[0]
        
        # Train unet
        with tf.GradientTape() as tape:
            prediction = self.unet(data[0])
           
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data[1], prediction),axis=(1,2)
                ),axis=(0,1)
            )

        if (train):
            grads = tape.gradient(loss, self.unet.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.unet.trainable_weights))
        self.loss_tracker.update_state(loss)
        
        return {
            "loss": self.loss_tracker.result(),
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        prediction = self.unet(input)
        return prediction