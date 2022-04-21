import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv


def get_reg(input_size,name="regressor"):
        
        if (input_size[0]==1): ## 2D image
            input_size = input_size[1:]
            
        if len(input_size) == 4:
            conv_layer = keras.layers.Conv3D
            
        else:
            conv_layer = keras.layers.Conv2D
            
        
        layer_dim = np.array(input_size[:-1],dtype=np.int32)
        filters = 16
        
        bf_input = keras.Input(shape=input_size,dtype=tf.float16)
        prediction_input = keras.Input(shape=input_size,dtype=tf.float16)
        x = keras.layers.Concatenate(axis=-1)([bf_input,prediction_input])
        ## downsampling layers    
        while layer_dim[0] > 1:
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_conv1_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn1_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.ReLU()(x)
            x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_conv2_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn2_{}".format(name,layer_dim[0]))(x)
            x =keras.layers.ReLU()(x)
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',name="{}_convd_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bnd_{}".format(name,layer_dim[0]))(x)            
            x =keras.layers.ReLU()(x)

        ## last conv same resulotion
        # x = conv_layer(filters=1,kernel_size=3,strides=1,padding='same',name="{}_convdense".format(name))(x)
        features = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256,activation="relu",dtype=tf.float32,name="{}_dense1".format(name))(features)
        x = keras.layers.Dense(64,activation="relu",dtype=tf.float32,name="{}_dense2".format(name))(x)
        output = keras.layers.Dense(1,activation="relu",dtype=tf.float32,name="{}_denseout".format(name))(x)
        
        return keras.Model([bf_input,prediction_input],[output,features],name=name)  

class RegCNN(keras.Model):
    def __init__(self, reg, **kwargs):
        super(RegCNN, self).__init__(**kwargs)

        self.reg = reg
        
        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )
        self.mae_loss_tracker = keras.metrics.Mean(
            name="mae_loss"
        )
        
    def pearson_corr(self,y_true, y_pred):
        x = y_true
        y = y_pred
        mean_x = tf.reduce_mean(x)
        mean_y = tf.reduce_mean(y)
        std_x = tf.math.reduce_std(x-mean_x)
        std_y = tf.math.reduce_std(y-mean_y)
        cc = tf.reduce_mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)

        return cc   
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.mae_loss_tracker,
        ]

    def train_step(self, data, train=True):
      
        # Train regressor
        with tf.GradientTape() as tape:
            prediction = (data[0][1])
            predicted_score, _ = self.reg([data[0][0],prediction])
            target_score = self.pearson_corr(data[1],prediction)
            loss = keras.losses.mean_squared_error(target_score, predicted_score)
            mae_loss = keras.losses.mean_absolute_error(target_score, predicted_score)
            

        if (train):
            grads = tape.gradient(loss, self.reg.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.reg.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.mae_loss_tracker.update_state(mae_loss)
        
        return {
            "loss": self.loss_tracker.result(),
            "mae_loss": self.mae_loss_tracker.result(),
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        return self.reg(input)
        