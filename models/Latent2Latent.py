import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv

def get_encoder(input_size,latent_dim,name="encoder"):
        
        if (input_size[0]==1): ## 2D image
            input_size = input_size[1:]
            
        if len(input_size) == 4:
            conv_layer = keras.layers.Conv3D
        else:
            conv_layer = keras.layers.Conv2D
        
        layer_dim = np.array(input_size[:-1],dtype=np.int32)
        filters = 8
        
        input = keras.Input(shape=input_size)
        
        ## first conv same resulotion
        x = conv_layer(filters=filters,kernel_size=4,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(input)
        if (gv.batch_norm):
            x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
        
        
        ## downsampling layers          
        while layer_dim[0] > 1:
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            x = conv_layer(filters=filters,kernel_size=4,strides=2,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
            

        ## flatten
        # x = conv_layer(filters=filters,kernel_size=1,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_1X1".format(name))(x)
        # if (gv.batch_norm):
        #     x = keras.layers.BatchNormalization(name="{}_bn_1X1".format(name))(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)
        
        
        ## z
        # x = keras.layers.Dense(512, name="{}_fc1".format(name))(x) ##check if improve
        z = keras.layers.Dense(latent_dim, name="{}_z".format(name),kernel_regularizer="l2")(x)
        
        return keras.Model(input,z,name=name), np.int32((*layer_dim,filters)), filters

    
class Latent2Latent(keras.Model):
    def __init__(self, input_encoder, target_encoder, target_decoder, **kwargs):
        super(Latent2Latent, self).__init__(**kwargs)
        
        self.input_encoder = input_encoder
        self.target_encoder = target_encoder
        self.target_decoder = target_decoder
        self.input_encoder_loss_tracker = keras.metrics.Mean(
            name="input_encoder_loss"
        )
    
    @property
    def metrics(self):
        return [
            self.input_encoder_loss_tracker,
        ]

    def train_step(self, data, train=True):
        data = tf.cast(data,dtype=tf.float16)

        # Train encoder
        with tf.GradientTape() as tape:
            z_pred = self.input_encoder(data[0])
            z_true = self.target_encoder(data[1])
            input_encoder_loss = keras.losses.mean_squared_error(z_true, z_pred)
        if (train):
            grads = tape.gradient(input_encoder_loss, self.input_encoder.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.input_encoder.trainable_weights))
        self.input_encoder_loss_tracker.update_state(input_encoder_loss)
        
        
        return {
            "input_encoder_loss":self.input_encoder_loss_tracker.result(),
        }
        
    def test_step(self, data):
        return self.train_step(data,False)
        
    def call(self,input):
        z = self.input_encoder(input)
        prediction = self.target_decoder(z)
        return prediction