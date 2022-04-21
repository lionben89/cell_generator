import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv


def get_pm(input_size,name="pair_matching"):
        
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
        x = keras.layers.Dense(64,activation="relu",dtype=tf.float32,name="{}_dense1".format(name))(features)
        # x = keras.layers.Dense(64,activation="relu",dtype=tf.float32,name="{}_dense2".format(name))(x)
        output = keras.layers.Dense(1,activation="sigmoid",dtype=tf.float32,name="{}_denseout".format(name))(x)
        
        return keras.Model([bf_input,prediction_input],output,name=name)  

class PMCNN(keras.Model):
    def __init__(self, pm, **kwargs):
        super(PMCNN, self).__init__(**kwargs)

        self.pm = pm
        
        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )
        self.acc = keras.metrics.BinaryAccuracy(
            name="acc"
        )
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.acc,
        ]

    def train_step(self, data, train=True):
      
        # Train regressor
        with tf.GradientTape() as tape:
            data_1 = tf.cast(data[1],dtype=tf.float32)
            input_shuffled = tf.random.shuffle(data[0])
            target_shuffled = tf.random.shuffle(data_1)
            pos_match_score = self.pm([data[0],data_1])
            neg_match_score = self.pm([input_shuffled,target_shuffled])
            target_score = tf.concat([tf.ones_like(pos_match_score),tf.zeros_like(neg_match_score)],axis=0)
            prediction_score = tf.concat([pos_match_score,neg_match_score],axis=0)
            
            loss = keras.losses.binary_crossentropy(target_score, prediction_score)

        if (train):
            grads = tape.gradient(loss, self.pm.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.pm.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        self.acc.update_state(target_score,prediction_score)
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc.result(),
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        return self.pm([input,input])
        