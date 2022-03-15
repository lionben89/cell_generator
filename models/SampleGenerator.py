import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv


class ReSampling(keras.layers.Layer):
    """Uses z to change 1 feature in each sample."""

    def call(self, z, sigma):
        updates = []
        indices=[]
        batch_size = gv.batch_size
        num_features = np.random.randint(0,16)
        for j in range(num_features):
            for i in range(batch_size):
                col = sample_col()
                change = sample_value(sigma)
                indices.append([i,col])
                updates.append(z[i,col]+change)
        
        z_new = tf.tensor_scatter_nd_update(z, indices, updates)
        return z_new

def sample_col():
    return np.random.randint(0,gv.latent_dim)


def sample_value(sigma):
    return np.random.uniform(-1.0*sigma,sigma)

def get_adaptor(input_size,layers=[32,32,32,32],name="adaptor"):
        
        if (input_size[0]==1): ## 2D image
            input_size = input_size[1:]
            
        if len(input_size) == 4:
            conv_layer = keras.layers.Conv3D
        else:
            conv_layer = keras.layers.Conv2D
        
        input = keras.Input(shape=input_size)

        x = input
        
        ## adaptor layers          
        for i in range(len(layers)):
            x = conv_layer(filters=layers[i],kernel_size=3,strides=1,padding='same',activation='relu',name="{}_conv_{}".format(name,i))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,i))(x)
        
        output = conv_layer(filters=1,kernel_size=3,strides=1,padding='same',activation='relu',name="{}_conv_out".format(name))(x)
        
        return keras.Model(input,output,name=name)

        
def get_latent_preprocessor(latent_dim,output_size,layer_dim,layers=[8,16,32,64],name="latent_preprocessor"):
        
        if (output_size[0]==1): ## 2D image
            output_size = output_size[1:]
        
        if len(output_size) == 4:
            conv_layer = keras.layers.Conv3DTranspose
            upsample_layer = keras.layers.UpSampling3D
        else:
            conv_layer = keras.layers.Conv2DTranspose
            upsample_layer = keras.layers.UpSampling2D
            
        input = keras.Input(shape=(latent_dim,))
        
        x =  input
        
        ## unflatten + dense layer
        x = keras.layers.Dense(tf.math.reduce_prod(layer_dim),name="{}_fc_1".format(name))(x)
        x = keras.layers.Reshape(layer_dim,name="{}_reshape".format(name))(x)
        
        ## preprocess layers  
        for i in range(len(layers)):
            x = conv_layer(filters=layers[i],kernel_size=3,strides=1,padding='same',activation='relu',name="{}_conv_{}".format(name,i))(x)
            x = upsample_layer(name="upsample_{}".format(i))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,i))(x)
      

        ## last conv same resulotion
        output = conv_layer(filters=32, kernel_size=3,strides=1,padding='same',activation='relu',name="{}_conv_{}_out".format(name,layer_dim[0]))(x)

        return keras.Model(input,output,name=name)  

def get_discriminator_image(input_size,name="discriminator_image"):
        
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
        x = conv_layer(filters=filters,kernel_size=3,strides=1,padding='same',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(input)
        if (gv.batch_norm):
            x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(input)
        
        
        ## downsampling layers          
        while layer_dim[0] > 1:
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            x = conv_layer(filters=filters,kernel_size=3,strides=2,padding='same',activation='relu',name="{}_conv_{}".format(name,layer_dim[0]))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
            

        ## flatten + dense layer
        x = keras.layers.Flatten()(x) #keras.layers.Dense(tf.math.reduce_prod(layer_dim), activation='relu', name="{}_fc_1".format(name))(keras.layers.Flatten()(x))
        x = keras.layers.Dense(128, activation="relu",name="{}_fc1".format(name))(x)
        
        ## output
        output = keras.layers.Dense(1, activation="sigmoid",name="{}_output".format(name))(x)
        
        return keras.Model(input,output,name=name)

   
class SampleGenerator(keras.Model):
    def __init__(self,latent_dim, patch_size, latent_preprocessor, adaptor, discriminator_image, unet, aae, **kwargs):
        super(SampleGenerator, self).__init__(**kwargs)

        self.discriminator_image = discriminator_image
        self.unet = unet
        self.aae = aae
        
        image_input = keras.layers.Input(shape=patch_size)
        latent_input = keras.layers.Input(shape=(latent_dim))
        processed_latent = latent_preprocessor(latent_input)
        processed_latent = keras.layers.Conv3D(filters=32,kernel_size=3,padding="same",activation="relu")(processed_latent)
        processed_image = keras.layers.Conv3D(filters=32,kernel_size=3,padding="same",activation="relu")(image_input)
        processed_input = keras.layers.Concatenate(axis=-1)([processed_image,processed_latent])
        output = adaptor(processed_input)
        self.generator = keras.Model([image_input,latent_input],output,name="generator")
        
        
        self.sigma = 0.0 #0.2
        self.sigma_count = 0
        self.resample = keras.Model(latent_input,ReSampling()(latent_input,self.sigma))
        
        self.unet_loss_tracker = keras.metrics.Mean(
            name="unet_loss"
        )
        
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.discriminator_image_loss_tracker = keras.metrics.Mean(name="discriminator_image_loss")
        self.discriminator_image_acc = keras.metrics.BinaryAccuracy(name="discriminator_image_accuracy", dtype=None,threshold=0.5)
         
        self.generator_loss_tracker = keras.metrics.Mean(
            name="generator_loss"
        )
        

    
    def compile(self, d_optimizer, g_optimizer, unet_loss_weight=1, discriminator_loss_weight=8, reconstruction_loss_weight=1):
        super(SampleGenerator, self).compile()
        self.d_image_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.unet_loss_weight = unet_loss_weight
        self.discriminator_loss_weight = discriminator_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
    
    @property
    def metrics(self):
        return [
            self.unet_loss_tracker,
            self.reconstruction_loss_tracker,
            self.discriminator_image_loss_tracker,
            self.discriminator_image_acc,
            self.generator_loss_tracker,
        ]

    def train_step(self, data, train=True):
        data_0 = tf.cast(data[0],dtype=tf.float16)
        # data_1 = tf.cast(data[1],dtype=tf.float16)
        
        unet_original = self.unet(data_0)
        z_original = self.aae.encoder(unet_original)
        
        z = self.resample(z_original,self.sigma)
        self.sigma_count+=1
        if (self.sigma_count>=20 and self.sigma<2.0):
            self.sigma+=0.1
            self.sigma_count=0
        
        # Train generator to fool discriminator_image and create target
        with tf.GradientTape() as tape:
            adapted_image = self.generator([data_0,z])
            discriminator_input = tf.concat([adapted_image,data_0],axis=0)
            generator_labels = tf.concat([tf.ones_like(z)[:,:1], tf.zeros_like(z)[:,:1]],axis=0)
            discriminator_predictions = self.discriminator_image(discriminator_input)
            generator_loss = keras.losses.binary_crossentropy(generator_labels, discriminator_predictions, from_logits=True)
            
            unet_predictions = self.unet(adapted_image)
            unet_target = self.aae.decoder(z)
            unet_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(unet_target, unet_predictions),axis=(2,3)
                )
            )
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data_0, adapted_image),axis=(2,3)
                )
            )
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.unet_loss_tracker.update_state(unet_loss)  
             
            total_loss = tf.add(tf.multiply(tf.cast(self.unet_loss_weight,dtype=tf.float16),unet_loss),tf.multiply(tf.cast(self.discriminator_loss_weight,dtype=tf.float16),generator_loss))
            total_loss = tf.add(total_loss,tf.multiply(tf.cast(self.reconstruction_loss_weight,dtype=tf.float16),reconstruction_loss))
            total_loss = tf.divide(total_loss,tf.add(tf.add(tf.cast(self.unet_loss_weight,dtype=tf.float16),tf.cast(self.discriminator_loss_weight,dtype=tf.float16)),tf.cast(self.reconstruction_loss_weight,dtype=tf.float16)))
            
            
        if (train):
            grads = tape.gradient(total_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.generator_loss_tracker.update_state(total_loss)   
        
        # Train discriminator_image for real/fake identification
        with tf.GradientTape() as tape:
            adapted_image = self.generator([data_0,z])
            discriminator_input = tf.concat([adapted_image,data_0],axis=0)
            discriminator_labels = tf.concat([tf.zeros_like(z)[:,:1], tf.ones_like(z)[:,:1]],axis=0)
            discriminator_predictions = self.discriminator_image(discriminator_input)
            discriminator_image_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(discriminator_labels, discriminator_predictions, from_logits=True),axis=0
            )
            
        if (train):
            grads = tape.gradient(discriminator_image_loss, self.discriminator_image.trainable_weights)
            self.d_image_optimizer.apply_gradients(zip(grads, self.discriminator_image.trainable_weights))
        self.discriminator_image_loss_tracker.update_state(discriminator_image_loss) 
        self.discriminator_image_acc.update_state(discriminator_labels,discriminator_predictions)
               
        return {
            "unet": self.unet_loss_tracker.result(),
            "reconstruction": self.reconstruction_loss_tracker.result(),
            "generator": self.generator_loss_tracker.result(),
            "discriminator": self.discriminator_image_loss_tracker.result(),
            "discriminator_acc": self.discriminator_image_acc.result(),
        }
        
    def test_step(self, data, train=True):
        return self.train_step(data,False)    
    
    def call(self,input):
        original_unet = self.unet(input)
        z = self.aae.encoder(original_unet)
        return self.generator([input,z])