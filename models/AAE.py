import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import sklearn as sklearn

def shuffle(x,y):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_x = tf.gather(x, shuffled_indices)
    shuffled_y = tf.gather(y, shuffled_indices)
    return shuffled_x,shuffled_y

def sample_distribution(batch_size,latent_dim, mu=0,sigma=1.):
    samples = mu + tf.random.normal(shape=(batch_size, latent_dim)) * sigma
    return samples

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
        x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(input)
        x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
        
        
        ## downsampling layers          
        while layer_dim[0] > 4:
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
            

        ## flatten + dense layer
        x = keras.layers.BatchNormalization(name="{}_bn_dense_{}".format(name,layer_dim[0]))(x)
        x = keras.layers.Flatten()(x) #keras.layers.Dense(tf.math.reduce_prod(layer_dim), activation='relu', name="{}_fc_1".format(name))(keras.layers.Flatten()(x))
        
        
        ## z
        z = keras.layers.Dense(latent_dim, name="{}_z".format(name))(x)
        
        return keras.Model(input,z,name=name), np.int32((*layer_dim,filters)), filters

        
def get_decoder(latent_dim,output_size,layer_dim,filters,name="decoder"):
        
        if (output_size[0]==1): ## 2D image
            output_size = output_size[1:]
        
        if len(output_size) == 4:
            conv_layer = keras.layers.Conv3DTranspose
        else:
            conv_layer = keras.layers.Conv2DTranspose
        
        input = keras.Input(shape=(latent_dim,))
        
        ## unflatten + dense layer
        x = keras.layers.Reshape(layer_dim)(keras.layers.Dense(tf.math.reduce_prod(layer_dim),name="{}_fc_1".format(name))(input))
        x = keras.layers.BatchNormalization(name="{}_bn_dense_{}".format(name,layer_dim[0]))(x)

        x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
        x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
        
        ## upsampling layers  
        while layer_dim[0] < output_size[0]:
            layer_dim = layer_dim * 2
            filters = filters // 2
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
      

        ## last conv same resulotion
        output = conv_layer(filters=1, kernel_size=3,strides=1,padding='same',kernel_initializer='glorot_normal',activation='relu',name="{}_conv_{}_out".format(name,layer_dim[0]))(x)

        return keras.Model(input,output,name=name)  

def get_discriminator(latent_dim,name="discriminator"):
        
        input = keras.Input(shape=(latent_dim,))
        x = keras.layers.Dense(1000, activation="relu",name="{}_fc1".format(name))(input)
        x = keras.layers.Dense(1000, activation="relu",name="{}_fc2".format(name))(x)
        output = keras.layers.Dense(1, activation="sigmoid",name="{}_output".format(name))(x)
        
        return keras.Model(input,output,name=name)
    
class AAE(keras.Model):
    def __init__(self, encoder, decoder, generator_input_size, discriminator, **kwargs):
        super(AAE, self).__init__(**kwargs)
        
        if (generator_input_size[0]==1): ## 2D image
            generator_input_size = generator_input_size[1:]
        generator_input = keras.Input(shape=generator_input_size)
        self.encoder = encoder
        self.decoder = decoder
        self.generator = keras.Model(generator_input,self.decoder(self.encoder(generator_input)),name="generator")
        
        self.discriminator = discriminator
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.generator_loss_tracker = keras.metrics.Mean(
            name="generator_loss"
        )
        self.discriminator_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
    
    def compile(self, d_optimizer, g_optimizer):
        super(AAE, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
    
    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
        ]

    def train_step(self, data):
        batch_size = tf.shape(data[0])[0]
        # Train generator for reconstruction
        with tf.GradientTape() as tape:
            z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data[1], reconstruction),axis=(1,2)
                )
            )
        # grads = tape.gradient(reconstruction_loss, [self.decoder.trainable_weights,self.encoder.trainable_weights])
        # self.g_optimizer.apply_gradients(zip(grads[0], self.decoder.trainable_weights))
        # self.g_optimizer.apply_gradients(zip(grads[1], self.encoder.trainable_weights))
        grads = tape.gradient(reconstruction_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        
        # Train discriminator for real/fake identification
        with tf.GradientTape() as tape:
            z = self.encoder(data[0])
            z_sample = sample_distribution(batch_size,z.shape[1])
            discriminator_input = tf.concat([z, z_sample],axis=0)
            discriminator_labels = tf.concat([tf.zeros_like(z)[:,:1], tf.ones_like(z)[:,:1]],axis=0)
            discriminator_input,discriminator_labels = shuffle(discriminator_input,discriminator_labels)
            discriminator_predictions = self.discriminator(discriminator_input)
            discriminator_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(discriminator_labels, discriminator_predictions, from_logits=True),axis=0
            )

        grads = tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        
        # Train generator to fool discriminator
        with tf.GradientTape() as tape:
            z = self.encoder(data[0])
            z_sample = sample_distribution(batch_size,z.shape[1])
            discriminator_input = tf.concat([z, z_sample],axis=0)
            generator_labels = tf.concat([tf.ones_like(z)[:,:1], tf.zeros_like(z)[:,:1]],axis=0)
            discriminator_input,generator_labels = shuffle(discriminator_input,generator_labels)
            discriminator_predictions = self.discriminator(discriminator_input)
            generator_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(generator_labels, discriminator_predictions, from_logits=True),axis=0
            )
            

        # grads = tape.gradient(generator_loss, [self.decoder.trainable_weights,self.encoder.trainable_weights])
        # self.g_optimizer.apply_gradients(zip(grads[0], self.decoder.trainable_weights))
        # self.g_optimizer.apply_gradients(zip(grads[1], self.encoder.trainable_weights))
        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.generator_loss_tracker.update_state(generator_loss)
        
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
            "generator_loss": self.generator_loss_tracker.result(),
        }
        
    def call(self,input):
        z = self.encoder(input)
        reconstruction = self.decoder(z)
        return reconstruction