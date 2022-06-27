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
        filters = 32
        
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
            x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_conv2_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn2_{}".format(name,layer_dim[0]))(x)
            x =keras.layers.ReLU()(x)
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
            x = convt_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_convt2_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bnt2_{}".format(name,layer_dim[0]))(x)                  
            x =keras.layers.ReLU()(x)
            filters = filters / 2
            # x = i_x + x
            i=i-1              
      

        ## last conv same resulotion
        output = conv_layer(filters=1, kernel_size=3,strides=1,padding='same',activation=activation,dtype=tf.float32,name="{}_conv_{}_out".format(name,layer_dim[0]))(x)

        return keras.Model(input,output,name=name)  

class UNET(keras.Model):
    def __init__(self, unet, pm, **kwargs):
        super(UNET, self).__init__(**kwargs)
        
        self.unet = unet
        self.pm = pm
        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )
        self.pm_loss_tracker = keras.metrics.Mean(
            name="pm_loss"
        )
        self.total_loss_tracker = keras.metrics.Mean(
            name="total_loss"
        )
        
        self.target_acc = keras.metrics.BinaryAccuracy()
        self.prediction_acc = keras.metrics.BinaryAccuracy()
        # self.acc_tracker = keras.metrics.MeanIoU(2,name="iou")
        
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.pm_loss_tracker,
            self.total_loss_tracker,
            self.target_acc,
            self.prediction_acc,
            # self.acc_tracker
        ]

    def train_step(self, data, train=True):
        # data = tf.cast(data,dtype=tf.float16)
        # batch_size = tf.shape(data[0])[0]
        
        # Train unet
        with tf.GradientTape() as tape:
            prediction = self.unet(data[0][0])
            # pm_prediction_target = self.pm(data[0])
            # pm_prediction_unet = self.pm([data[0][0],prediction])
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data[0][1], prediction),axis=(1,2,3)
                ),axis=0
            )
            pm_loss = 0 #keras.losses.mean_absolute_error(pm_prediction_target, pm_prediction_unet)
            total_loss = 0.25 * loss + 1*pm_loss
            

        if (train):
            grads = tape.gradient(total_loss, self.unet.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.unet.trainable_weights))
        # self.loss_tracker.update_state(loss)
        # self.pm_loss_tracker.update_state(pm_loss)
        self.total_loss_tracker.update_state(total_loss)
        # self.target_acc.update_state(tf.ones_like(pm_prediction_target),pm_prediction_target)
        # self.prediction_acc.update_state(tf.ones_like(pm_prediction_unet),pm_prediction_unet)
        # self.acc_tracker.update_state(data[1], tf.math.round(prediction))
        
        
        return {
            # "loss": self.loss_tracker.result(),
            # "pm_loss": self.pm_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            # "target_acc":self.target_acc.result(),
            # "pred_acc":self.prediction_acc.result()
            # "iou": self.acc_tracker.result()
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        prediction = self.unet(input[0])
        return prediction