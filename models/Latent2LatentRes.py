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
        x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(input)
        if (gv.batch_norm):
            x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
        
        
        ## downsampling layers          
        while layer_dim[0] > 1:
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
            

        ## flatten + dense layer
        x = keras.layers.Flatten()(x) #keras.layers.Dense(tf.math.reduce_prod(layer_dim), activation='relu', name="{}_fc_1".format(name))(keras.layers.Flatten()(x))
        
        
        ## z
        z = keras.layers.Dense(latent_dim, name="{}_z".format(name))(x)
        
        return keras.Model(input,z,name=name), np.int32((*layer_dim,filters)), filters
    
def get_adaptor(input_size,name="adaptor"):
        
        if (input_size[0]==1): ## 2D image
            input_size = input_size[1:]
            conv_layer = keras.layers.Conv2D  
        else:
            conv_layer = keras.layers.Conv3D    
        
        input_size = (*input_size[:-1],input_size[-1])
        
        input = keras.Input(shape=input_size)
        x=input
        
        ## downsampling layers          
        i=0
        filters=32
        while i < 4:
            # filters = filters * 2
            x = conv_layer(filters=filters,kernel_size=4,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_adaptor_{}".format(name,i))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn_adaptor_{}".format(name,i))(x)
            i+=1

        ## conv out
        output = conv_layer(filters=1,kernel_size=4,strides=1,padding='same',kernel_initializer='glorot_normal',activation='sigmoid',name="{}_conv_out".format(name))(x)
        
        return keras.Model(input,output,name=name)

    
class Latent2LatentRes(keras.Model):
    def __init__(self, input_encoder, adaptor, target_encoder, target_decoder, **kwargs):
        super(Latent2LatentRes, self).__init__(**kwargs)
        
        self.input_encoder = input_encoder
        self.adaptor = adaptor
        self.target_encoder = target_encoder
        self.target_decoder = target_decoder
        
        self.input_encoder_loss_tracker = keras.metrics.Mean(
            name="input_encoder_loss"
        )
        self.adaptor_loss_tracker = keras.metrics.Mean(
            name="adaptor_loss"
        )
    
    @property
    def metrics(self):
        return [
            self.input_encoder_loss_tracker,
            self.adaptor_loss_tracker
        ]
        
    def compile(self, encoder_optimizer, adaptor_optimizer):
        super(Latent2LatentRes, self).compile()
        self.e_optimizer = encoder_optimizer
        self.a_optimizer = adaptor_optimizer

    def train_step(self, data, train=True):
        data = tf.cast(data,dtype=tf.float16)

        # Train encoder
        with tf.GradientTape() as tape:
            z_pred = self.input_encoder(data[0])
            z_true = self.target_encoder(data[1])
            input_encoder_loss = keras.losses.mean_squared_error(z_true, z_pred)
        if (train):    
            grads = tape.gradient(input_encoder_loss, self.input_encoder.trainable_weights)
            self.e_optimizer.apply_gradients(zip(grads, self.input_encoder.trainable_weights))
        self.input_encoder_loss_tracker.update_state(input_encoder_loss)
            
        # Train adaptor
        with tf.GradientTape() as tape:
            z_pred = self.input_encoder(data[0])
            temp_prediction = self.target_decoder(z_pred)
            # adaptor_input = tf.concat([temp_prediction,data[0]],axis=-1)
            prediction = self.adaptor(temp_prediction)
            adaptor_loss = keras.losses.mean_absolute_error(data[1], prediction)        
        
        if (train):
            grads = tape.gradient(adaptor_loss, self.adaptor.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.adaptor.trainable_weights))
        self.adaptor_loss_tracker.update_state(adaptor_loss)
        
        
        return {
            "input_encoder_loss":self.input_encoder_loss_tracker.result(),
            "adaptor_loss":self.adaptor_loss_tracker.result(),
            "total_loss":self.input_encoder_loss_tracker.result()+self.adaptor_loss_tracker.result()
        }
    
    def test_step(self, data):
        return self.train_step(data, False)
        
    def call(self,input):
        z = self.input_encoder(input)
        temp_prediction = self.target_decoder(z)
        # adaptor_input = tf.concat([temp_prediction,input],axis=-1)
        prediction = self.adaptor(temp_prediction)
        return prediction