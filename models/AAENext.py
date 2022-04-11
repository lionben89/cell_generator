from json import encoder
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import global_vars as gv

def shuffle(x,y):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_x = tf.gather(x, shuffled_indices)
    shuffled_y = tf.gather(y, shuffled_indices)
    return shuffled_x,shuffled_y

def sample_distribution(batch_size,latent_dim, mu=0,sigma=1.):
    samples = mu + tf.random.normal(shape=(batch_size, latent_dim),dtype=tf.float16) * sigma
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
        x=input
        x = conv_layer(filters=filters,kernel_size=4,strides=1,padding='same',name="{}_conv_{}".format(name,layer_dim[0]))(x)
        
        ## downsampling layers          
        while layer_dim[0] > 1:
            i_x = x
            x = conv_layer(filters=filters,kernel_size=7,strides=1,padding='same',name="{}_conv1_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.Dense(4*filters)(x)
            x = keras.layers.Activation("gelu")(x)
            x = keras.layers.Dense(filters)(x)
            # x = conv_layer(filters=filters,kernel_size=7,strides=1,padding='same',name="{}_conv2_{}".format(name,layer_dim[0]))(x)
            # x = keras.layers.Dense(4*filters)(x)
            # x = keras.layers.Activation("gelu")(x)
            # x = keras.layers.Dense(filters)(x)
            # x = conv_layer(filters=filters,kernel_size=7,strides=1,padding='same',name="{}_conv3_{}".format(name,layer_dim[0]))(x)
            # x = keras.layers.Dense(4*filters)(x)
            # x = keras.layers.Activation("gelu")(x)
            # x = keras.layers.Dense(filters)(x)
            x = i_x+x
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            x = conv_layer(filters=filters,kernel_size=2,strides=2,padding='same',name="{}_convd2_{}".format(name,layer_dim[0]))(x)
                
            # x = conv_layer(filters=filters,kernel_size=7,strides=1,padding='same',activation="relu",kernel_initializer='glorot_normal',name="{}_conv2_{}".format(name,layer_dim[0]))(x)
            # if (gv.batch_norm):
            #     x = keras.layers.LayerNormalization(name="{}_bn2_{}".format(name,layer_dim[0]))(x)                
            

        ## z cnn
        # x = conv_layer(filters=1,kernel_size=1,strides=(1,2,2),padding='same',kernel_initializer='glorot_normal',name="{}_convout".format(name))(x)
        # z = keras.layers.Flatten()(x) #keras.layers.Dense(tf.math.reduce_prod(layer_dim), activation='relu', name="{}_fc_1".format(name))(keras.layers.Flatten()(x))
        # return keras.Model(input,z,name=name), np.int32((1,8,8,1)), filters    
        
        ## z dense
        # x = conv_layer(filters=512,kernel_size=7,strides=1,padding='same',kernel_initializer='glorot_normal',name="{}_convout".format(name))(x)
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(latent_dim*2,activation="relu",  name="{}_fc1".format(name))(x)
        z = keras.layers.Dense(latent_dim, name="{}_z".format(name))(x)
        return keras.Model(input,z,name=name), np.int32((*layer_dim,filters)), filters
    

        
def get_decoder(latent_dim,output_size,layer_dim,filters, activation="relu", name="decoder"):
        
        if (output_size[0]==1): ## 2D image
            output_size = output_size[1:]
        
        if len(output_size) == 4:
            conv_layer = keras.layers.Conv3DTranspose
        else:
            conv_layer = keras.layers.Conv2DTranspose
        
        input = keras.Input(shape=(latent_dim,))
        x=input
        ## unflatten + dense layer
        x = keras.layers.Dense(tf.math.reduce_prod(layer_dim),name="{}_fc_1".format(name))(input)
        # x = keras.layers.Dense(tf.math.reduce_prod(layer_dim)*2,activation="relu",name="{}_fc_2".format(name))(x)
        x = keras.layers.Reshape(layer_dim,name="{}_reshape".format(name))(x)
        
        # x = conv_layer(filters=filters,kernel_size=7,strides=(1,2,2),padding='same',activation="relu",kernel_initializer='glorot_normal',name="{}_convt_{}".format(name,layer_dim[0]))(x)
        # if (gv.batch_norm):
        #     x = keras.layers.LayerNormalization(name="{}_bn_{}".format(name,layer_dim[0]))(x)
            
        x = conv_layer(filters=filters,kernel_size=4,strides=1,padding='same',name="{}_convt_{}".format(name,layer_dim[0]))(x)
        
        ## upsampling layers  
        while layer_dim[0] < output_size[0]:
            i_x = x
            x = conv_layer(filters=filters,kernel_size=7,strides=1,padding='same',name="{}_convt1_{}".format(name,layer_dim[0]))(x)
            x = keras.layers.Dense(4*filters)(x)
            x = keras.layers.Activation("gelu")(x)
            x = keras.layers.Dense(filters)(x)
            # x = conv_layer(filters=filters,kernel_size=7,strides=1,padding='same',name="{}_convt2_{}".format(name,layer_dim[0]))(x)
            # x = keras.layers.Dense(4*filters)(x)
            # x = keras.layers.Activation("gelu")(x)
            # x = keras.layers.Dense(filters)(x)
            # x = conv_layer(filters=filters,kernel_size=7,strides=1,padding='same',name="{}_convt3_{}".format(name,layer_dim[0]))(x)
            # x = keras.layers.Dense(4*filters)(x)
            # x = keras.layers.Activation("gelu")(x)
            # x = keras.layers.Dense(filters)(x)                        
            x = i_x+x               
            layer_dim = layer_dim * 2
            filters = filters // 2
            x = conv_layer(filters=filters,kernel_size=2,strides=2,padding='same',name="{}_convtu2_{}".format(name,layer_dim[0]))(x)
                
            # x = conv_layer(filters=filters,kernel_size=7,strides=1,padding='same',activation="relu",kernel_initializer='glorot_normal',name="{}_convt2_{}".format(name,layer_dim[0]))(x)
            # if (gv.batch_norm):
            #     x = keras.layers.LayerNormalization(name="{}_bn2_{}".format(name,layer_dim[0]))(x)
      

        ## last conv same resulotion
        output = conv_layer(filters=1, kernel_size=7, strides=1, activation=activation, padding='same',name="{}_conv_{}_out".format(name,layer_dim[0]))(x)
        return keras.Model(input,output,name=name)  

def get_discriminator_latent(latent_dim,name="discriminator_latent"):
        
        input = keras.Input(shape=(latent_dim,))
        x = keras.layers.Dense(512, activation="relu",name="{}_fc1".format(name))(input)
        x = keras.layers.Dense(512, activation="relu",name="{}_fc2".format(name))(x)
        # x = keras.layers.Dense(128, activation="relu",name="{}_fc3".format(name))(x)
        output = keras.layers.Dense(1, activation="sigmoid",name="{}_output".format(name))(x)
        
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
        x = input
        x = conv_layer(filters=filters,kernel_size=7,strides=1,padding='same',activation="relu",kernel_initializer='glorot_normal',name="{}_conv_{}".format(name,layer_dim[0]))(input)
        
        
        ## downsampling layers          
        while layer_dim[0] > 1:
            layer_dim = np.int32(layer_dim / 2)
            filters = filters * 2
            x = conv_layer(filters=filters,kernel_size=7,strides=2,padding='same',activation="relu",kernel_initializer='glorot_normal',name="{}_conv_{}".format(name,layer_dim[0]))(x)

        ## flatten + dense layer
        x = conv_layer(filters=64,kernel_size=7,strides=1,padding='same',activation="relu",kernel_initializer='glorot_normal',name="{}_convout".format(name))(x)
        x = keras.layers.Flatten()(x) #keras.layers.Dense(tf.math.reduce_prod(layer_dim), activation='relu', name="{}_fc_1".format(name))(keras.layers.Flatten()(x))
        x = keras.layers.Dense(512, activation="relu",name="{}_fc1".format(name))(x)
        
        ## output
        output = keras.layers.Dense(1, activation="sigmoid",name="{}_output".format(name))(x)
        
        return keras.Model(input,output,name=name)
    
class AAE(keras.Model):
    def __init__(self, encoder, decoder, generator_input_size, discriminator_latent, discriminator_image, **kwargs):
        super(AAE, self).__init__(**kwargs)
        
        if (generator_input_size[0]==1): ## 2D image
            generator_input_size = generator_input_size[1:]
        generator_input = keras.Input(shape=generator_input_size)
        self.encoder = encoder
        self.decoder = decoder
        self.generator = keras.Model(generator_input,self.decoder(self.encoder(generator_input)),name="generator")
        
        self.discriminator_latent = discriminator_latent
        self.discriminator_image = discriminator_image
        
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction"
        )
        self.generator_loss_tracker = keras.metrics.Mean(
            name="generator"
        )
        self.encoder_loss_tracker = keras.metrics.Mean(
            name="encoder"
        )
        self.discriminator_latent_loss_tracker = keras.metrics.Mean(name="d_latent")
        self.discriminator_latent_acc = keras.metrics.BinaryAccuracy(name="d_latent_acc", dtype=None,threshold=0.5)
        self.discriminator_image_loss_tracker = keras.metrics.Mean(name="d_image")
        self.discriminator_image_acc = keras.metrics.BinaryAccuracy(name="d_image_acc", dtype=None,threshold=0.5)
        # self.acc_tracker = keras.metrics.MeanIoU(2,name="iou")
        
    
    def compile(self, d_latent_optimizer, d_image_optimizer, g_optimizer):
        super(AAE, self).compile()
        self.d_latent_optimizer = d_latent_optimizer
        self.d_image_optimizer = d_image_optimizer
        self.g_optimizer = g_optimizer
    
    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
            self.generator_loss_tracker,
            self.encoder_loss_tracker,
            self.discriminator_latent_loss_tracker,
            self.discriminator_image_loss_tracker,
            self.discriminator_latent_acc,
            self.discriminator_image_acc,
            # self.acc_tracker
        ]

    def train_step(self, data, train=True):
        data = tf.cast(data,dtype=tf.float16)
        batch_size = tf.shape(data[0])[0]
        
        # Train generator to fool discriminator_image
        with tf.GradientTape() as tape:
            z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            # reconstruction_pixelwise_weight = tf.where(data[1]>0,tf.zeros_like(data[1])+0.5,tf.zeros_like(data[1])+0.01)
            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         tf.math.multiply(0.1*tf.square(data[1]-reconstruction),reconstruction_pixelwise_weight),axis=(1,2,3)
            #     ),axis=0
            # )
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data[1], reconstruction),axis=(1,2)
                ),axis=(0,1)
            )
            
            # reconstruction_loss = tf.reduce_mean(
            #         keras.losses.mean_squared_error(data[1], reconstruction)
            #     ,axis=(0,1,2,3)
            # )
            
            # self.acc_tracker.update_state(data[1], tf.math.round(reconstruction))
            # reconstruction_loss =  tf.cast(reconstruction_loss,dtype=tf.float16)
            # reconstruction_loss = 10*tf.reduce_mean(keras.losses.binary_crossentropy(data[1], reconstruction),axis=(0,1,2,3))
            
            # discriminator_input = tf.concat([reconstruction,data[1]],axis=0)
            # generator_labels = tf.concat([tf.ones_like(z)[:,:1], tf.zeros_like(z)[:,:1]],axis=0)
            # # discriminator_input,generator_labels = shuffle(discriminator_input,generator_labels)
            # discriminator_predictions = self.discriminator_image(discriminator_input)
            # generator_loss = tf.reduce_mean(keras.losses.binary_crossentropy(generator_labels, discriminator_predictions, from_logits=True),axis=0)
            
            # self.generator_loss_tracker.update_state(generator_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            
            # total_loss = tf.add(tf.multiply(encoder_loss,tf.constant(1,dtype=tf.float16)),tf.add(tf.multiply(reconstruction_loss,tf.constant(0.1,dtype=tf.float16)),tf.multiply(generator_loss,tf.constant(1,dtype=tf.float16))))
            total_loss = reconstruction_loss #+ generator_loss
            
        if (train):
            grads = tape.gradient(total_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Train discriminator_image for real/fake identification
        # with tf.GradientTape() as tape:
        #     z = self.encoder(data[0])
        #     reconstruction = self.decoder(z)
        #     discriminator_input = tf.concat([reconstruction,data[1]],axis=0)
        #     discriminator_labels = tf.concat([tf.zeros_like(z)[:,:1], tf.ones_like(z)[:,:1]],axis=0)
        #     discriminator_input,discriminator_labels = shuffle(discriminator_input,discriminator_labels)
        #     discriminator_predictions = self.discriminator_image(discriminator_input)
        #     discriminator_image_loss = tf.reduce_mean(
        #         keras.losses.binary_crossentropy(discriminator_labels, discriminator_predictions, from_logits=True),axis=0
        #     )
        # if (train):
        #     grads = tape.gradient(discriminator_image_loss, self.discriminator_image.trainable_weights)
        #     self.d_image_optimizer.apply_gradients(zip(grads, self.discriminator_image.trainable_weights))
        # self.discriminator_image_loss_tracker.update_state(discriminator_image_loss) 
        # self.discriminator_image_acc.update_state(discriminator_labels,discriminator_predictions)
        
        # Train discriminator_latent for real/fake identification
        with tf.GradientTape() as tape:
            z = self.encoder(data[0])
            z_sample = sample_distribution(batch_size,z.shape[1])
            discriminator_input = tf.concat([z, z_sample],axis=0)
            discriminator_labels = tf.concat([tf.zeros_like(z)[:,:1], tf.ones_like(z)[:,:1]],axis=0)
            discriminator_input,discriminator_labels = shuffle(discriminator_input,discriminator_labels)
            discriminator_predictions = self.discriminator_latent(discriminator_input)
            discriminator_latent_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(discriminator_labels, discriminator_predictions, from_logits=True),axis=0
            )
                     
        if (train):
            grads = tape.gradient(discriminator_latent_loss, self.discriminator_latent.trainable_weights)
            self.d_latent_optimizer.apply_gradients(zip(grads, self.discriminator_latent.trainable_weights))
        self.discriminator_latent_loss_tracker.update_state(discriminator_latent_loss)
        self.discriminator_latent_acc.update_state(discriminator_labels,discriminator_predictions)
        
        # Train generator_latent for real/fake identification
        with tf.GradientTape() as tape:
            z = self.encoder(data[0])     
            z_sample = sample_distribution(batch_size,z.shape[1])
            discriminator_input = tf.concat([z, z_sample],axis=0)
            generator_labels = tf.concat([tf.ones_like(z)[:,:1], tf.zeros_like(z)[:,:1]],axis=0)
            # discriminator_input,generator_labels = shuffle(discriminator_input,generator_labels)
            discriminator_predictions = self.discriminator_latent(discriminator_input)
            encoder_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(generator_labels, discriminator_predictions, from_logits=True),axis=0
            )
            
            if (train):
                grads = tape.gradient(encoder_loss, self.encoder.trainable_weights)
                self.g_optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))
            self.encoder_loss_tracker.update_state(encoder_loss)   
        
        return {
            "reconstruction": self.reconstruction_loss_tracker.result(),
            "d_latent": self.discriminator_latent_loss_tracker.result(),
            # "d_image": self.discriminator_image_loss_tracker.result(),
            # "generator": self.generator_loss_tracker.result(),
            "g_latent":self.encoder_loss_tracker.result(),
            "d_latent_acc": self.discriminator_latent_acc.result(),
            # "d_image_acc": self.discriminator_image_acc.result(),
            # "iou": self.acc_tracker.result()
        }
        
    def test_step(self, data):
        return self.train_step(data,False)    
    
    def call(self,input):
        z = self.encoder(input)
        reconstruction = self.decoder(z)
        return reconstruction