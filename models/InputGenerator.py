import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv


def get_adaptor(input_size,layers=[64,64,64,64],name="adaptor"):
        
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

   
class InputGenerator(keras.Model):
    def __init__(self, patch_size, adaptor, discriminator_image, unet, **kwargs):
        super(InputGenerator, self).__init__(**kwargs)

        self.discriminator_image = discriminator_image
        self.unet = unet
        
        image_input = keras.layers.Input(shape=patch_size)
        target_input = keras.layers.Input(shape=patch_size)
        processed_target = keras.layers.Conv3D(filters=32,kernel_size=3,padding="same",activation="relu")(target_input)
        processed_image = keras.layers.Conv3D(filters=32,kernel_size=3,padding="same",activation="relu")(image_input)
        processed_input = keras.layers.Concatenate(axis=-1)([processed_image,processed_target])
        output = adaptor(processed_input)
        self.generator = keras.Model([image_input,target_input],output,name="generator")
        
        self.unet_loss_tracker = keras.metrics.Mean(
            name="unet_loss"
        )
        
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.discriminator_image_loss_tracker = keras.metrics.Mean(name="discriminator_image_loss")
        self.discriminator_image_acc = keras.metrics.BinaryAccuracy(name="discriminator_image_accuracy", dtype=None,threshold=0.5)
        
        self.total_loss_tracker = keras.metrics.Mean(name = "total")
        

    
    def compile(self, d_optimizer, g_optimizer, unet_loss_weight=3, discriminator_loss_weight=0, reconstruction_loss_weight=1):
        super(InputGenerator, self).compile()
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
        data_0 = data[0] #tf.cast(data[0],dtype=tf.float16)
        data_1 = data[1]# data_1 = tf.cast(data[1],dtype=tf.float16)
        
        # input_shuffled = tf.random.shuffle(data_0)
        prediction = self.unet(data_0)
        
        # Train generator to fool discriminator_image and create target
        with tf.GradientTape() as tape:
            adapted_image = self.generator([data_0,prediction])
            # discriminator_input = tf.concat([adapted_image,data_0],axis=0)
            # generator_labels = tf.concat([tf.ones_like(adapted_image)[:,:1,0,0,0], tf.zeros_like(adapted_image)[:,:1,0,0,0]],axis=0)
            # discriminator_predictions = self.discriminator_image(discriminator_input)
            # generator_loss = tf.reduce_mean(keras.losses.binary_crossentropy(generator_labels, discriminator_predictions),axis=0)
            
            unet_predictions = self.unet(adapted_image)
            
            unet_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data_1, unet_predictions),axis=(1,2)
                ),axis=(0,1)
            )
            # unet_loss = keras.losses.mean_squared_error(target_shuffled, unet_predictions)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data_0, adapted_image),axis=(1,2)
                ),axis=(0,1)
            )
            # reconstruction_loss = keras.losses.mean_squared_error(data_0, adapted_image)
            
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.unet_loss_tracker.update_state(unet_loss)  
             
            total_loss =  reconstruction_loss*self.reconstruction_loss_weight + unet_loss*self.unet_loss_weight #+ generator_loss*self.discriminator_loss_weight
            total_loss = total_loss / (self.discriminator_loss_weight + self.unet_loss_weight + self.reconstruction_loss_weight)
            self.total_loss_tracker.update_state(total_loss)
            
        if (train):
            grads = tape.gradient(total_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights)) 
        
        # # Train discriminator_image for real/fake identification
        # with tf.GradientTape() as tape:
        #     discriminator_input = tf.concat([adapted_image,data_0],axis=0)
        #     discriminator_labels = tf.concat([tf.zeros_like(adapted_image)[:,:1,0,0,0], tf.ones_like(adapted_image)[:,:1,0,0,0]],axis=0)
        #     discriminator_predictions = self.discriminator_image(discriminator_input)
        #     discriminator_image_loss = tf.reduce_mean(
        #         keras.losses.binary_crossentropy(discriminator_labels, discriminator_predictions),axis=0
        #     )
            
        # if (train):
        #     grads = tape.gradient(discriminator_image_loss, self.discriminator_image.trainable_weights)
        #     self.d_image_optimizer.apply_gradients(zip(grads, self.discriminator_image.trainable_weights))
        # self.discriminator_image_loss_tracker.update_state(discriminator_image_loss) 
        # self.discriminator_image_acc.update_state(discriminator_labels,discriminator_predictions)
               
        return {
            "unet": self.unet_loss_tracker.result(),
            "reconstruction": self.reconstruction_loss_tracker.result(),
            # "discriminator": self.discriminator_image_loss_tracker.result(),
            # "discriminator_acc": self.discriminator_image_acc.result(),
            "total_loss":self.total_loss_tracker.result()
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        original_unet = self.unet(input)
        return self.generator([input,original_unet])