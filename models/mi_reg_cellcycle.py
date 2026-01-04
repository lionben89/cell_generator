import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from metrics import tf_pearson_corr
from callbacks import *
import os
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Ensure eager execution (TF2 does this by default)
tf.compat.v1.enable_eager_execution()

###############--LOAD--##########################
load = False
train = True
adabatch = False

# Number of samples to visualize
num_samples = 100
num_samples_subset = 500


class MaskInterpreterRegression(keras.Model):
    def __init__(self, patch_size, adaptor, regressor, pcc_target=0.9, **kwargs):
        """
        Mask Interpreter adapted for regression tasks.
        
        Args:
            patch_size (tuple): Original input image shape, e.g. (64, 64, 1)
            adaptor (keras.Model): A network that, given processed inputs, outputs a mask.
            regressor (keras.Model): A pretrained regression model (e.g., for cell cycle markers).
            pcc_target (float): Target PCC value between the regressor's predictions on
                               the original and perturbed images. Defaults to 0.9.
        """
        super(MaskInterpreterRegression, self).__init__(**kwargs)
        self.pcc_target = pcc_target
        self.regressor = regressor  # this regressor is assumed fixed
        
        # Augment input with gradient: new channel count = original channels + 1.
        augmented_input_shape = (patch_size[0], patch_size[1], patch_size[2] + 1)
        image_input = keras.layers.Input(shape=augmented_input_shape, dtype=tf.float32)
        
        # Pass the augmented input directly to the adaptor
        # The adaptor expects (H, W, C+1) where C=1, so (64, 64, 2)
        importance_mask = tf.cast(adaptor(image_input), dtype=tf.float64)
        self.generator = keras.Model(image_input, importance_mask, name="generator")
        
        # ---- Define metrics ----
        self.similarity_loss_tracker = keras.metrics.Mean(name="similarity_loss")
        self.binary_size_mask = keras.metrics.BinaryAccuracy(name="binary_size_mask")
        self.importance_mask_size = keras.metrics.Mean(name="importance_mask_size")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.pcc = keras.metrics.Mean(name="pcc")
        self.stop = keras.metrics.Mean(name="stop")

    def compile(self, g_optimizer, similarity_loss_weight=1.0, mask_loss_weight=1.0,
                noise_scale=1.0, target_loss_weight=0.0, run_eagerly=False):
        """
        Args:
            g_optimizer: Optimizer for updating the adaptor network.
            similarity_loss_weight (float): Weight for the similarity loss.
            mask_loss_weight (float): Weight for the mask loss.
            noise_scale (float): Standard deviation of the noise applied.
            target_loss_weight (float): Weight for the PCC target loss.
            run_eagerly (bool): Whether to run eagerly.
        """
        super(MaskInterpreterRegression, self).compile(run_eagerly=run_eagerly)
        self.g_optimizer = g_optimizer
        self.similarity_loss_weight = similarity_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.noise_scale = noise_scale
        self.target_loss_weight = target_loss_weight

    @property
    def metrics(self):
        return [
            self.similarity_loss_tracker,
            self.total_loss_tracker,
            self.binary_size_mask,
            self.importance_mask_size,
            self.pcc,
            self.stop,
        ]
    
    def _augment_input_with_gradients(self, x):
        """
        Compute the gradient of the regressor's output with respect to the input x
        and use its magnitude as an additional channel.
        
        Args:
            x: Tensor of shape (B, H, W, C) (e.g. (B, 64, 64, 1))
        Returns:
            augmented_x: Tensor of shape (B, H, W, C+1)
        """
        x_float = tf.cast(x, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_float)
            preds = self.regressor(x_float)  # shape (B, 1) for regression
            # Use the regression output directly
            output = tf.reduce_sum(preds, axis=1)  # shape (B,)
        
        # Compute gradients of the output with respect to x
        grad = tape.gradient(output, x_float)  # shape (B, H, W, C)
        
        # Compute the L2 norm of the gradients along the channel dimension
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=-1, keepdims=True))  # shape (B, H, W, 1)
        
        # Normalize grad_norm to [0,1] per sample
        epsilon = 1e-8
        min_val = tf.reduce_min(grad_norm, axis=[1,2,3], keepdims=True)
        max_val = tf.reduce_max(grad_norm, axis=[1,2,3], keepdims=True)
        grad_norm = (grad_norm - min_val) / (max_val - min_val + epsilon)
        
        # Concatenate the original input with the grad_norm as an extra channel
        augmented_x = tf.concat([x, tf.cast(grad_norm, x.dtype)], axis=-1)  # shape (B, H, W, C+1)
        return augmented_x

    def train_step(self, data):
        # Expect data as input images
        x = tf.cast(data, dtype=tf.float64)  # original input, shape (B, H, W, C)
        
        # Compute regressor output on the original image (for loss computation)
        regressor_target = tf.cast(self.regressor(x), dtype=tf.float64)
        
        # Augment input with gradient channel
        augmented_x = self._augment_input_with_gradients(tf.cast(x, tf.float32))
        augmented_x = tf.cast(augmented_x, tf.float32)
        
        with tf.GradientTape() as tape:
            # Generate importance mask from the augmented input
            importance_mask = self.generator(augmented_x)  # shape (B, H, W, 1)
            
            # Create random noise
            normal_noise = tf.random.normal(tf.shape(importance_mask), stddev=self.noise_scale, dtype=tf.float64)
            
            # Compute adapted (noisy) image: use importance_mask to mix x and noise
            adapted_image = (importance_mask * x) + (normal_noise * (1 - importance_mask))
            
            # Get regressor's output on the adapted image
            regressor_output = tf.cast(self.regressor(tf.cast(adapted_image, tf.float32)), dtype=tf.float64)
            
            # Compute similarity loss (MSE between regressor outputs on original and adapted images)
            similarity_loss = tf.reduce_mean(tf.square(regressor_target - regressor_output))
            
            # Regularization on the mask (encourage sparsity)
            mean_importance_mask = tf.reduce_mean(importance_mask)
            mask_loss = tf.reduce_mean(tf.abs(importance_mask))
            
            # Compute PCC between regressor outputs
            pcc_value = tf_pearson_corr(regressor_target, regressor_output)
            pcc_loss = tf.abs(self.pcc_target - pcc_value)
            
            # Total loss
            total_loss = (similarity_loss * self.similarity_loss_weight) + \
                         (mask_loss * self.mask_loss_weight) + \
                         (pcc_loss * self.target_loss_weight)
        
        grads = tape.gradient(total_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Update metrics
        self.similarity_loss_tracker.update_state(similarity_loss)
        self.importance_mask_size.update_state(1 - mean_importance_mask)
        self.binary_size_mask.update_state(tf.zeros_like(importance_mask), importance_mask)
        self.pcc.update_state(pcc_value)
        self.total_loss_tracker.update_state(total_loss)
        self.stop.update_state(pcc_loss + mean_importance_mask)
        
        return {
            "similarity_loss": self.similarity_loss_tracker.result(),
            "binary_size": self.binary_size_mask.result(),
            "importance_mask_size": self.importance_mask_size.result(),
            "total_loss": self.total_loss_tracker.result(),
            "pcc": self.pcc.result(),
            "stop": self.stop.result()
        }
    
    def test_step(self, data):
        return self.train_step(data)
    
    def call(self, inputs):
        # For inference, augment the input with gradient channel and produce the mask
        augmented_x = self._augment_input_with_gradients(tf.cast(inputs, tf.float32))
        return self.generator(tf.cast(augmented_x, tf.float32))
    
    def predict_from_noisy(self, inputs):
        # Given a single image, augment it with gradient channel, compute adapted image and return regressor prediction
        augmented_x = self._augment_input_with_gradients(tf.cast(np.expand_dims(inputs, axis=0), tf.float32))
        mask = self.generator.predict(augmented_x)
        mask = np.squeeze(mask)  # shape becomes (H,W) or (H,W,1)
        
        if mask.ndim == 2:
            mask_expanded = np.expand_dims(mask, axis=-1)
        else:
            mask_expanded = mask
            
        noise = tf.random.normal(shape=inputs.shape, stddev=self.noise_scale, dtype=tf.float64).numpy()
        adapted_image = (mask_expanded * inputs.astype(np.float64)) + ((1 - mask_expanded) * noise)
        pred_value_adapted = self.regressor.predict(np.expand_dims(adapted_image.astype(np.float32), axis=0))
        return pred_value_adapted


###############--DATA LOADING--##########################

def load_single_sample(img_file, labels_dir):
    """Load a single image-label pair. Used for parallel processing."""
    try:
        filename = os.path.basename(img_file)
        label_file = os.path.join(labels_dir, filename)
        
        if not os.path.exists(label_file):
            return None, None
            
        # Load image: T, C, Y, X - take first timepoint and channel
        bf = np.load(img_file)[0, 0]  # Shape: (64, 64)
        
        # Load labels: C, T, 1 - take first timepoint for both channels
        target = np.load(label_file)[:, 0, 0]  # Shape: (2,) - [marker1, marker2]
        
        return bf, target
    except Exception as e:
        print(f"Error loading {img_file}: {e}")
        return None, None

def load_cell_cycle_data(data_dir, num_workers=None):
    """Load all cell cycle data from the given directory using parallel processing."""
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
    
    print(f"  Found {len(image_files)} files in {data_dir}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"  Loading with {num_workers} workers...")
    
    # Parallel loading with progress bar
    load_func = partial(load_single_sample, labels_dir=labels_dir)
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(load_func, image_files),
            total=len(image_files),
            desc="  Loading files",
            unit="file"
        ))
    
    # Filter out failed loads and separate images and labels
    all_images = []
    all_labels = []
    
    for bf, target in results:
        if bf is not None and target is not None:
            all_images.append(bf)
            all_labels.append(target)
    
    # Convert to numpy arrays
    images = np.array(all_images)
    labels = np.array(all_labels)
    
    # Add channel dimension for grayscale images
    images = np.expand_dims(images, axis=-1)  # Shape: (N, Y, X, 1)
    
    print(f"  Loaded {len(images)} samples")
    print(f"  Image shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return images, labels


if __name__ == "__main__":
    # Load pre-split datasets
    base_data_dir = "/groups/assafza_group/assafza/Gad/Cell_Cycle_Data"
    
    print("\nLoading training set...")
    x_train, y_train = load_cell_cycle_data(os.path.join(base_data_dir, "train"))
    
    print("\nLoading validation set...")
    x_val, y_val = load_cell_cycle_data(os.path.join(base_data_dir, "val"))
    
    print("\nLoading test set...")
    x_test, y_test = load_cell_cycle_data(os.path.join(base_data_dir, "test"))
    
    # Normalize images using dataset-specific statistics
    print("\nNormalizing images...")
    x_train = x_train.astype("float32")
    x_val = x_val.astype("float32")
    x_test = x_test.astype("float32")
    
    # Compute mean and std for each dataset separately
    mean_train = np.mean(x_train)
    std_train = np.std(x_train)
    mean_val = np.mean(x_val)
    std_val = np.std(x_val)
    mean_test = np.mean(x_test)
    std_test = np.std(x_test)
    
    print(f"Training set - Mean: {mean_train:.4f}, Std: {std_train:.4f}")
    print(f"Validation set - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"Test set - Mean: {mean_test:.4f}, Std: {std_test:.4f}")
    
    # Apply normalization
    x_train = (x_train - mean_train) / std_train
    x_val = (x_val - mean_val) / std_val
    x_test = (x_test - mean_test) / std_test
    
    ###############--LOAD TRAINED REGRESSORS--##########################
    
    print("\nLoading trained regression models...")
    marker1_model = tf.keras.models.load_model('cellcycle_marker1.h5')
    marker1_model.trainable = False
    
    marker2_model = tf.keras.models.load_model('cellcycle_marker2.h5')
    marker2_model.trainable = False
    
    ###############--DEFINE ADAPTOR--##########################
    
    from models.UNETO import *
    # Adaptor for 64x64 grayscale images with 2 channels (image + gradient)
    adaptor_m1 = get_unet((64, 64, 2), activation="sigmoid")
    adaptor_m1.summary()
    
    adaptor_m2 = get_unet((64, 64, 2), activation="sigmoid")
    
    ###############--CREATE MASK INTERPRETERS--##########################
    
    print("\n" + "="*60)
    print("Creating Mask Interpreter for Marker 1...")
    print("="*60)
    
    mi_marker1 = MaskInterpreterRegression(
        patch_size=(64, 64, 1),
        adaptor=adaptor_m1,
        regressor=marker1_model,
        pcc_target=0.95
    )
    
    mi_marker1.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        similarity_loss_weight=1.0,
        mask_loss_weight=1.0,
        noise_scale=0.5,
        target_loss_weight=1.75
    )
    
    # Build the model
    mi_marker1(np.random.random((1, 64, 64, 1)).astype(np.float32))
    
    print("\n" + "="*60)
    print("Creating Mask Interpreter for Marker 2...")
    print("="*60)
    
    mi_marker2 = MaskInterpreterRegression(
        patch_size=(64, 64, 1),
        adaptor=adaptor_m2,
        regressor=marker2_model,
        pcc_target=0.95
    )
    
    mi_marker2.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        similarity_loss_weight=1.0,
        mask_loss_weight=1.0,
        noise_scale=0.5,
        target_loss_weight=1.75
    )
    
    # Build the model
    mi_marker2(np.random.random((1, 64, 64, 1)).astype(np.float32))
    
    ###############--SETUP CALLBACKS--##########################
    
    model_path_m1 = "./cellcycle_mi_marker1"
    model_path_m2 = "./cellcycle_mi_marker2"
    
    checkpoint_callback_m1 = SaveModelCallback(
        1, mi_marker1, model_path_m1, 
        monitor="val_stop", term="val_pcc", term_value=1.0
    )
    
    checkpoint_callback_m2 = SaveModelCallback(
        1, mi_marker2, model_path_m2,
        monitor="val_stop", term="val_pcc", term_value=1.0
    )
    
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_stop',
        patience=7,
        restore_best_weights=True
    )
    
    ###############--TRAINING--##########################
    
    if load:
        print("\nLoading pre-trained Mask Interpreters...")
        mi_marker1.load_weights(model_path_m1)
        mi_marker2.load_weights(model_path_m2)
    
    if train:
        print("\n" + "="*60)
        print("Training Mask Interpreter for Marker 1...")
        print("="*60)
        
        mi_marker1.fit(
            x_train,
            epochs=200,
            batch_size=128,
            callbacks=[early_stop_callback, checkpoint_callback_m1],
            validation_data=(x_val, None)
        )
        
        print("\n" + "="*60)
        print("Training Mask Interpreter for Marker 2...")
        print("="*60)
        
        mi_marker2.fit(
            x_train,
            epochs=200,
            batch_size=128,
            callbacks=[early_stop_callback, checkpoint_callback_m2],
            validation_data=(x_val, None)
        )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
