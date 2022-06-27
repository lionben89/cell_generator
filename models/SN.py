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
        filters = 64
        
        input = keras.Input(shape=input_size,dtype=tf.float16)
        x = input
        ## downsampling layers    
        while layer_dim[0] > 2:
            layer_dim = np.int32(layer_dim / 2)
            # filters = filters * 2
            x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_conv1_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn1_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.ReLU()(x)
            # x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',name="{}_conv2_{}".format(name,layer_dim[0]))(x)
            # if (gv.batch_norm):
            #     x = keras.layers.BatchNormalization(name="{}_bn2_{}".format(name,layer_dim[0]))(x)
            # x =keras.layers.ReLU()(x)
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',name="{}_convd_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bnd_{}".format(name,layer_dim[0]))(x)            
            x =keras.layers.ReLU()(x)

        ## last conv same resulotion
        # x = conv_layer(filters=1,kernel_size=3,strides=1,padding='same',name="{}_convdense".format(name))(x)
        features = keras.layers.Flatten()(x)
        output = keras.layers.Dense(256,activation="tanh", dtype=tf.float32,name="{}_dense1".format(name))(features)
        # x = keras.layers.Dense(64,activation="relu",dtype=tf.float32,name="{}_dense2".format(name))(x)
        # output = keras.layers.Dense(1,activation="sigmoid",dtype=tf.float32,name="{}_denseout".format(name))(x)
        
        return keras.Model(input,output,name=name)  

def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    x,y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

class SN(keras.Model):
    def __init__(self,input_size, pm1,pm2,**kwargs):
        super(SN, self).__init__(**kwargs)

        self.pm1 = pm1
        self.pm2 = pm2

        
        input1 = keras.Input(shape=input_size,dtype=tf.float16)
        input2 = keras.Input(shape=input_size,dtype=tf.float16)
        f1 = self.pm1(input1)
        f2 = self.pm2(input2)
        ec = keras.layers.Lambda(euclidean_distance,name="ec")([f1, f2])
        ec = keras.layers.BatchNormalization(name="bn")(ec)
        output = keras.layers.Dense(1,activation="sigmoid",dtype=tf.float32)(ec)
        self.model = keras.Model([input1,input2],output,name="SN")
        
        self.loss_tracker = keras.metrics.Mean()
        self.ba_tracker = keras.metrics.BinaryAccuracy()
        self.recall_tracker = keras.metrics.Recall()
        self.precision_tracker = keras.metrics.Precision()
    

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.ba_tracker,
            self.recall_tracker,
            self.precision_tracker
        ]

    def train_step(self, data, train=True):
      
        # Train regressor
        with tf.GradientTape() as tape:
            margin=1.0
            y_true = tf.cast(data[1],dtype=tf.float32)
            y_pred = self.model(data[0])
            loss = tf.reduce_mean(tf.reduce_sum(y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(tf.math.maximum(margin - y_pred, 0.0)),axis=-1))

        if (train):
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        self.ba_tracker.update_state(y_true,y_pred)
        self.recall_tracker.update_state(y_true,y_pred)
        self.precision_tracker.update_state(y_true,y_pred)

        return {
            "loss": self.loss_tracker.result(),
            "acc": self.ba_tracker.result(),
            "recall": self.recall_tracker.result(),
            "precision": self.precision_tracker.result(),
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        return self.model(input)
        