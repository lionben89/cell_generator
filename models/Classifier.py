import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv

"""Simple Classifier"""
def get_clf(input_size, num_classes ,name="classifier"):
        
        if (input_size[0]==1): ## 2D image
            input_size = input_size[1:]
            
        if len(input_size) == 4:
            conv_layer = keras.layers.Conv3D
            
        else:
            conv_layer = keras.layers.Conv2D
            
        
        layer_dim = np.array(input_size[:-1],dtype=np.int32)
        filters = 32
        
        input = keras.Input(shape=input_size,dtype=tf.float16)
        x = input
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
        output = keras.layers.Dense(num_classes,activation="softmax",dtype=tf.float32,name="{}_denseout".format(name))(x)
        
        return keras.Model(input,output,name=name)  
        