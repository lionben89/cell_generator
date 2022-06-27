import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv


def get_adaptor(input_size=[1,16,16,1],layers=[512,256,128,64],name="adaptor"):
            
        conv_layer = keras.layers.Conv3DTranspose
        
        input = keras.Input(shape=[64])

        x = keras.layers.Dense(256,activation="relu")(input)
        x = keras.layers.Reshape(input_size)(x)
        
        ## adaptor layers          
        for i in range(len(layers)):
            x = conv_layer(filters=layers[i],kernel_size=3,strides=2,padding='same',activation='relu',name="{}_conv_{}".format(name,i))(x)
            if (gv.batch_norm):
                x = keras.layers.BatchNormalization(name="{}_bn_{}".format(name,i))(x)
        
        output = conv_layer(filters=1,kernel_size=3,strides=1,padding='same',activation='relu',dtype=tf.float32,name="{}_conv_out".format(name))(x)
        
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
        x = keras.layers.Dense(512, activation="relu",name="{}_fc1".format(name))(x)
        
        ## output
        output = keras.layers.Dense(1, activation="sigmoid",dtype=tf.float32,name="{}_output".format(name))(x)
        
        return keras.Model(input,output,name=name)

   
class ShuffleGenerator(keras.Model):
    def __init__(self, adaptor, discriminator_image, unet, **kwargs):
        super(ShuffleGenerator, self).__init__(**kwargs)

        self.discriminator_image = discriminator_image
        self.unet = unet
        
        latent_input = keras.layers.Input(shape=[64])
        output = adaptor(latent_input)
        self.generator = keras.Model(latent_input,output,name="generator")
        
        self.unet_loss_tracker = keras.metrics.Mean(
            name="unet_loss"
        )
        
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.discriminator_image_loss_tracker = keras.metrics.Mean(name="discriminator_image_loss")
        self.discriminator_image_acc = keras.metrics.BinaryAccuracy(name="discriminator_image_accuracy", dtype=None,threshold=0.5)
        
        self.total_loss_tracker = keras.metrics.Mean(name = "total")
        self.image = None
    
    def compile(self, d_optimizer, g_optimizer, unet_loss_weight=0.0, discriminator_loss_weight=1, reconstruction_loss_weight=0.000):
        super(ShuffleGenerator, self).compile()
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
            self.total_loss_tracker
        ]

    def train_step(self, data, train=True):
         #tf.cast(data[0],dtype=tf.float16)
        # data_1 = tf.cast(data[1],dtype=tf.float16)
        batch_size = tf.shape(data[0])[0]
        data_0 = tf.random.normal([batch_size,64])#+tf.reduce_mean(data[0])
        
        # input_shuffled = tf.experimental.numpy.flip(data_0) ##tf.random.shuffle(data_0)
        unet_target = self.unet(data[0])
        
        # Train generator to fool discriminator_image and create target
        with tf.GradientTape() as tape:
            adapted_image = self.generator(data_0)
            unet_predictions = self.unet(adapted_image)
            discriminator_input = tf.concat([unet_predictions,unet_target],axis=0)
            generator_labels = tf.concat([tf.ones_like(adapted_image)[:,:1,0,0,0], tf.zeros_like(adapted_image)[:,:1,0,0,0]],axis=0)
            discriminator_predictions = self.discriminator_image(discriminator_input)
            generator_loss = tf.reduce_mean(keras.losses.binary_crossentropy(generator_labels, discriminator_predictions),axis=0)
            
            
            
            unet_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(unet_target, unet_predictions),axis=(1,2,3)
                ),axis=0
            )
            
            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         keras.losses.mean_squared_error(data_0, adapted_image),axis=(1,2,3)
            #     ),axis=0
            # )
            # self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.unet_loss_tracker.update_state(unet_loss)  
             
            total_loss =   generator_loss#unet_loss*self.unet_loss_weight #generator_loss*self.discriminator_loss_weight + reconstruction_loss*self.reconstruction_loss_weight +
            # total_loss = total_loss / (self.discriminator_loss_weight + self.unet_loss_weight + self.reconstruction_loss_weight)
            self.total_loss_tracker.update_state(total_loss)
            
        if (train):
            grads = tape.gradient(total_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights)) 

        #Train discriminator_image for real/fake identification
        with tf.GradientTape() as tape:
            discriminator_input = tf.concat([unet_predictions,unet_target],axis=0)
            discriminator_labels = tf.concat([tf.zeros_like(adapted_image)[:,:1,0,0,0], tf.ones_like(adapted_image)[:,:1,0,0,0]],axis=0)
            discriminator_predictions = self.discriminator_image(discriminator_input)
            discriminator_image_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(discriminator_labels, discriminator_predictions),axis=0
            )
            
        if (train):
            grads = tape.gradient(discriminator_image_loss, self.discriminator_image.trainable_weights)
            self.d_image_optimizer.apply_gradients(zip(grads, self.discriminator_image.trainable_weights))
        self.discriminator_image_loss_tracker.update_state(discriminator_image_loss) 
        self.discriminator_image_acc.update_state(discriminator_labels,discriminator_predictions)
               
        return {
            "unet": self.unet_loss_tracker.result(),
            "reconstruction": self.reconstruction_loss_tracker.result(),
            "discriminator": self.discriminator_image_loss_tracker.result(),
            "discriminator_acc": self.discriminator_image_acc.result(),
            "total loss":self.total_loss_tracker.result()
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        batch_size = tf.shape(input)[0]
        data_0 = tf.random.normal([batch_size,64])
        return self.generator(data_0)