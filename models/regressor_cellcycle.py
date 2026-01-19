"""
Cell Cycle Regression Model
===========================

This script trains ResNet18-based regression models to predict cell cycle markers
(Cdt1 and Geminin) from brightfield microscopy images of cells.

Overview:
---------
The script performs the following steps:
    1. Load cell cycle dataset (train/val/test/perturbations splits)
    2. Normalize images using per-dataset statistics
    3. Build and train two separate ResNet18-based regression models:
       - Marker 1 model: Predicts Cdt1 intensity (cell cycle G1/S marker)
       - Marker 2 model: Predicts Geminin intensity (cell cycle S/G2/M marker)
    4. Evaluate models on test and perturbation datasets
    5. Generate visualization plots (scatter, residual, distribution plots)

Data Format:
------------
    Images (.npy files):
        - Shape: (T, C, Y, X)
        - T = number of timepoints (typically 1)
        - C = number of channels/cell indices per image
        - Y, X = spatial dimensions (e.g., 64x64 pixels)
    
    Labels (.npy files):
        - Shape: (C, T, N)
        - C = number of marker channels (2: Cdt1, Geminin)
        - T = number of timepoints
        - N = number of cell indices (matches image C dimension)

Data Loading Modes:
-------------------
    The load_cell_cycle_data() function supports two modes:
    
    1. Load ALL indices (default):
       >>> x_train, y_train = load_cell_cycle_data(data_dir)
       This loads every cell index from each image file, useful for
       maximum data utilization.
    
    2. Load specific index:
       >>> x_train, y_train = load_cell_cycle_data(data_dir, index=0)
       This loads only index 0 from each image file, useful for
       consistent sampling or debugging.

Model Architecture:
-------------------
    ResNet18-based architecture with custom modifications:
    
    1. Custom Stem (for grayscale input):
       - Conv2D(64, 7x7, stride=2) -> BatchNorm -> ReLU -> MaxPool
    
    2. ResNet18 Backbone:
       - Block 1: 2x BasicBlock(64 filters)
       - Block 2: 2x BasicBlock(128 filters, first with stride=2)
       - Block 3: 2x BasicBlock(256 filters, first with stride=2)
       - Block 4: 2x BasicBlock(512 filters, first with stride=2)
       - Global Average Pooling -> 512-dim feature vector
    
    3. MLP Head:
       - Dense(512) embedding layer
       - 4x [Dense(512) -> LayerNorm -> GELU] blocks
       - Dense(1, sigmoid) output layer
    
    Output: Single value in [0, 1] representing marker intensity

Adaptive Batch Normalization:
-----------------------------
    When adabatch=True (default), batch normalization statistics are
    adapted to each evaluation dataset before predictions. This helps
    account for domain shift between training and test/perturbation data.
    
    The adapt_batch_norm() function runs forward passes with training=True
    to update running mean/variance statistics in BN layers.

Training Configuration:
-----------------------
    - Optimizer: Adam (learning_rate=1e-4)
    - Loss: Mean Squared Error (MSE)
    - Metrics: Mean Absolute Error (MAE)
    - Batch size: 32
    - Max epochs: 100
    - Early stopping: patience=10, restore_best_weights=True

Output Files:
-------------
    Models:
        - cellcycle_marker1.h5: Trained Cdt1 prediction model
        - cellcycle_marker2.h5: Trained Geminin prediction model
    
    Visualizations:
        - cellcycle_markers_results.png: Scatter and residual plots (test set)
        - cellcycle_cdt1_distribution.png: Cdt1 prediction distribution
        - cellcycle_geminin_distribution.png: Geminin prediction distribution
        - cellcycle_joint_distribution.png: Joint Cdt1-Geminin density (predicted)
        - cellcycle_joint_distribution_true.png: Joint density (ground truth)
        - cellcycle_markers_perturbations_results.png: Perturbation results
        - cellcycle_test_vs_perturbations_comparison.png: MAE comparison bar chart

Configuration Flags:
--------------------
    load (bool): If True, load pre-trained models instead of training.
                 Default: False (train new models)
    
    adabatch (bool): If True, apply adaptive batch normalization before
                     evaluation on each dataset. Default: True

Usage Example:
--------------
    # Train new models
    $ python cellcycle-clf.py
    
    # To load existing models instead of training, set load=True in the script

Dependencies:
-------------
    - tensorflow / keras
    - numpy
    - matplotlib
    - multiprocessing
    - tqdm
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

load = False
adabatch = True  # Enable adaptive batch normalization for test/perturbation sets

# 1. Load Cell Cycle Dataset
print("Loading Cell Cycle data...")

def load_single_sample(img_file, labels_dir, index=None):
    """Load image-label pairs from a file. Used for parallel processing.
    
    Args:
        img_file: Path to the image file
        labels_dir: Directory containing label files
        index: If None, load all indices. If int, load only that index.
    
    Returns:
        If index is None: tuple of (list of images, list of labels)
        If index is int: tuple of (single image, single label)
    """
    try:
        filename = os.path.basename(img_file)
        label_file = os.path.join(labels_dir, filename)
        
        if not os.path.exists(label_file):
            return None, None
        
        # Load full arrays
        img_data = np.load(img_file)  # Shape: (T, C, Y, X)
        label_data = np.load(label_file)  # Shape: (C, T, N)
        
        if index is None:
            # Load all indices
            num_indices = img_data.shape[1]  # C dimension
            images = []
            labels = []
            for idx in range(num_indices):
                # Image: T, C, Y, X - take first timepoint and current index
                bf = img_data[0, idx]  # Shape: (Y, X)
                # Labels: C, T, N - take all channels, first timepoint, current index
                target = label_data[:, 0, idx]  # Shape: (2,)
                images.append(bf)
                labels.append(target)
            return images, labels
        else:
            # Load single index (original behavior)
            bf = img_data[0, index]  # Shape: (64, 64)
            target = label_data[:, 0, index]  # Shape: (2,) - [marker1, marker2]
            return bf, target
    except Exception as e:
        print(f"Error loading {img_file}: {e}")
        return None, None

def load_cell_cycle_data(data_dir, num_workers=None, index=None):
    """Load all cell cycle data from the given directory using parallel processing.
    
    Args:
        data_dir: Directory containing images/ and labels/ subdirectories
        num_workers: Number of parallel workers (default: cpu_count - 1)
        index: If None, load all indices from each image. If int, load only that index.
    
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
    
    print(f"  Found {len(image_files)} files in {data_dir}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"  Loading with {num_workers} workers...")
    
    # Parallel loading with progress bar
    load_func = partial(load_single_sample, labels_dir=labels_dir, index=index)
    
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
            if index is None:
                # bf and target are lists when loading all indices
                all_images.extend(bf)
                all_labels.extend(target)
            else:
                # bf and target are single arrays when loading specific index
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

# Load pre-split datasets from train/val/test directories
base_data_dir = "/groups/assafza_group/assafza/Gad/Cell_Cycle_Data"

print("\nLoading training set...")
x_train, y_train = load_cell_cycle_data(os.path.join(base_data_dir, "train"))

print("\nLoading validation set...")
x_val, y_val = load_cell_cycle_data(os.path.join(base_data_dir, "val"))

print("\nLoading test set...")
x_test, y_test = load_cell_cycle_data(os.path.join(base_data_dir, "test"))

print("\nLoading perturbations set...")
x_pert, y_pert = load_cell_cycle_data(os.path.join(base_data_dir, "perturbations"))

# Normalize images using training set statistics
print("\nNormalizing images...")
x_train = x_train.astype("float32")
x_val = x_val.astype("float32")
x_test = x_test.astype("float32")
x_pert = x_pert.astype("float32")

# Compute mean and std for each dataset separately
mean_train = np.mean(x_train)
std_train = np.std(x_train)
print(f"Training set - Mean: {mean_train:.4f}, Std: {std_train:.4f}")

mean_val = np.mean(x_val)
std_val = np.std(x_val)
print(f"Validation set - Mean: {mean_val:.4f}, Std: {std_val:.4f}")

mean_test = np.mean(x_test)
std_test = np.std(x_test)
print(f"Test set - Mean: {mean_test:.4f}, Std: {std_test:.4f}")

mean_pert = np.mean(x_pert)
std_pert = np.std(x_pert)
print(f"Perturbations set - Mean: {mean_pert:.4f}, Std: {std_pert:.4f}")

# Apply normalization to each set using its own statistics
x_train = (x_train - mean_train) / std_train
x_val = (x_val - mean_val) / std_val
x_test = (x_test - mean_test) / std_test
x_pert = (x_pert - mean_pert) / std_pert

print(f"\nDataset summary:")
print(f"  Training set shape: {x_train.shape}")
print(f"  Validation set shape: {x_val.shape}")
print(f"  Test set shape: {x_test.shape}")
print(f"  Perturbations set shape: {x_pert.shape}")

# Get input shape from data
input_shape = x_train.shape[1:]
print(f"Input shape: {input_shape}")

def adapt_batch_norm(model, data, batch_size=32):
    """
    Adapt batch normalization statistics to new data.
    Sets BN layers to training mode to update running statistics.
    
    Args:
        model: Keras model with BatchNormalization layers
        data: Data to adapt to (numpy array)
        batch_size: Batch size for adaptation
    """
    print(f"  Adapting batch normalization statistics on {len(data)} samples...")
    
    # Find all BatchNormalization layers
    bn_layers = [layer for layer in model.layers if isinstance(layer, layers.BatchNormalization)]
    
    if not bn_layers:
        print("  Warning: No BatchNormalization layers found in model.")
        return
    
    # Create a temporary model that runs in training mode for BN layers only
    # We'll do a forward pass with training=True to update BN statistics
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]
        
        # Forward pass with training=True to update BN running statistics
        _ = model(batch, training=True)
    
    print(f"  Adapted {len(bn_layers)} BatchNormalization layers")

# 2. Build regression model for Marker 1
print("\n" + "="*60)
print("Building Marker 1 Regression Model...")
print("="*60)

def create_regression_model(input_shape, name="marker_model"):
    """Create a ResNet18-based regression model for cell cycle marker prediction.
    
    Architecture inspired by the original PyTorch model:
    - ResNet18 backbone (modified for grayscale input)
    - MLP head with LayerNorm and GELU activations
    - Outputs single value for regression
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Option 1: Replicate single channel to 3 channels (like original model)
    # x = layers.Concatenate()([inputs, inputs, inputs])  # (64, 64, 3)
    
    # Option 2: Use single channel directly with modified ResNet18
    # We'll use single channel since it's more efficient
    
    # Load ResNet18 base (without top, without pretrained weights)
    # Note: TensorFlow's ResNet expects 3 channels, so we'll build custom stem
    
    # Custom stem for grayscale input (replacing ResNet's first conv)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', 
                      use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # ResNet18 blocks (manually constructed)
    # Block 1: 64 channels, no downsampling
    x = _resnet_block(x, 64, stride=1, name='block1a')
    x = _resnet_block(x, 64, stride=1, name='block1b')
    
    # Block 2: 128 channels, downsample
    x = _resnet_block(x, 128, stride=2, name='block2a')
    x = _resnet_block(x, 128, stride=1, name='block2b')
    
    # Block 3: 256 channels, downsample
    x = _resnet_block(x, 256, stride=2, name='block3a')
    x = _resnet_block(x, 256, stride=1, name='block3b')
    
    # Block 4: 512 channels, downsample
    x = _resnet_block(x, 512, stride=2, name='block4a')
    x = _resnet_block(x, 512, stride=1, name='block4b')
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)  # -> (batch, 512)
    
    # MLP Head (like the original PyTorch model)
    # Embedding layer
    x = layers.Dense(512, name='head_embed')(x)
    
    # 4 MLP blocks with LayerNorm and GELU
    for i in range(4):
        x = layers.Dense(512, name=f'head_block{i}_dense')(x)
        x = layers.LayerNormalization(name=f'head_block{i}_ln')(x)
        x = layers.Activation('gelu', name=f'head_block{i}_gelu')(x)
    
    # Final projection to output
    outputs = layers.Dense(1, activation='sigmoid', name='head_proj')(x)
    
    model = keras.Model(inputs, outputs, name=name)
    return model

def _resnet_block(x, filters, stride=1, name='block'):
    """Create a ResNet basic block (used in ResNet18/34)."""
    shortcut = x
    
    # First conv
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same',
                      use_bias=False, name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)
    
    # Second conv
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                      use_bias=False, name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    
    # Shortcut connection (with projection if needed)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride,
                                 padding='same', use_bias=False,
                                 name=f'{name}_shortcut_conv')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
    
    # Add shortcut and apply ReLU
    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.Activation('relu', name=f'{name}_relu2')(x)
    
    return x

# Create and compile Marker 1 model
marker1_model = create_regression_model(input_shape, name="marker1_model")
marker1_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='mse',
    metrics=['mae']
)
marker1_model.summary()

# Early stopping callback
early_stopping_m1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

save_path_m1 = "cellcycle_marker1.h5"

if not load:
    print("\nTraining Marker 1 model...")
    history_m1 = marker1_model.fit(
        x_train, y_train[:, 0],  # Marker 1 is first column
        epochs=100,
        batch_size=32,
        validation_data=(x_val, y_val[:, 0]),
        callbacks=[early_stopping_m1],
        verbose=1
    )
    marker1_model.save(save_path_m1)
else:
    marker1_model = tf.keras.models.load_model(save_path_m1)

print(f"Marker 1 model saved to {save_path_m1}")

# 3. Build regression model for Marker 2
print("\n" + "="*60)
print("Building Marker 2 Regression Model...")
print("="*60)

marker2_model = create_regression_model(input_shape, name="marker2_model")
marker2_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='mse',
    metrics=['mae']
)

# Early stopping callback
early_stopping_m2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

save_path_m2 = "cellcycle_marker2.h5"

if not load:
    print("\nTraining Marker 2 model...")
    history_m2 = marker2_model.fit(
        x_train, y_train[:, 1],  # Marker 2 is second column
        epochs=100,
        batch_size=32,
        validation_data=(x_val, y_val[:, 1]),
        callbacks=[early_stopping_m2],
        verbose=1
    )
    marker2_model.save(save_path_m2)
else:
    marker2_model = tf.keras.models.load_model(save_path_m2)

print(f"Marker 2 model saved to {save_path_m2}")

# 4. Evaluate both models
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# Apply adaptive batch normalization if enabled
if adabatch:
    print("\nApplying adaptive batch normalization...")
    print("Marker 1 model:")
    adapt_batch_norm(marker1_model, x_val, batch_size=32)
    print("Marker 2 model:")
    adapt_batch_norm(marker2_model, x_val, batch_size=32)

# Marker 1 Evaluation
print("\nMarker 1 Results:")
print("-" * 40)
m1_train_loss, m1_train_mae = marker1_model.evaluate(x_train, y_train[:, 0], verbose=0)
print(f"Training   - MSE: {m1_train_loss:.6f}, MAE: {m1_train_mae:.6f}")

m1_val_loss, m1_val_mae = marker1_model.evaluate(x_val, y_val[:, 0], verbose=0)
print(f"Validation - MSE: {m1_val_loss:.6f}, MAE: {m1_val_mae:.6f}")

m1_test_loss, m1_test_mae = marker1_model.evaluate(x_test, y_test[:, 0], verbose=0)
print(f"Test       - MSE: {m1_test_loss:.6f}, MAE: {m1_test_mae:.6f}")

# Marker 2 Evaluation
print("\nMarker 2 Results:")
print("-" * 40)
m2_train_loss, m2_train_mae = marker2_model.evaluate(x_train, y_train[:, 1], verbose=0)
print(f"Training   - MSE: {m2_train_loss:.6f}, MAE: {m2_train_mae:.6f}")

m2_val_loss, m2_val_mae = marker2_model.evaluate(x_val, y_val[:, 1], verbose=0)
print(f"Validation - MSE: {m2_val_loss:.6f}, MAE: {m2_val_mae:.6f}")

m2_test_loss, m2_test_mae = marker2_model.evaluate(x_test, y_test[:, 1], verbose=0)
print(f"Test       - MSE: {m2_test_loss:.6f}, MAE: {m2_test_mae:.6f}")

# Perturbations Evaluation
print("\n" + "="*60)
print("PERTURBATIONS RESULTS")
print("="*60)

# Apply adaptive batch normalization for perturbations if enabled
if adabatch:
    print("\nApplying adaptive batch normalization for perturbations...")
    print("Marker 1 model:")
    adapt_batch_norm(marker1_model, x_pert, batch_size=32)
    print("Marker 2 model:")
    adapt_batch_norm(marker2_model, x_pert, batch_size=32)

# Marker 1 on Perturbations
print("\nMarker 1 Results (Perturbations):")
print("-" * 40)
m1_pert_loss, m1_pert_mae = marker1_model.evaluate(x_pert, y_pert[:, 0], verbose=0)
print(f"Perturbations - MSE: {m1_pert_loss:.6f}, MAE: {m1_pert_mae:.6f}")

# Marker 2 on Perturbations
print("\nMarker 2 Results (Perturbations):")
print("-" * 40)
m2_pert_loss, m2_pert_mae = marker2_model.evaluate(x_pert, y_pert[:, 1], verbose=0)
print(f"Perturbations - MSE: {m2_pert_loss:.6f}, MAE: {m2_pert_mae:.6f}")

# Compare test vs perturbations performance
print("\n" + "="*60)
print("PERFORMANCE COMPARISON: Test vs Perturbations")
print("="*60)
print(f"\nMarker 1 (Cdt1):")
print(f"  Test MAE:          {m1_test_mae:.6f}")
print(f"  Perturbations MAE: {m1_pert_mae:.6f}")
print(f"  Difference:        {m1_pert_mae - m1_test_mae:.6f} ({((m1_pert_mae/m1_test_mae - 1)*100):.2f}%)")

print(f"\nMarker 2 (Geminin):")
print(f"  Test MAE:          {m2_test_mae:.6f}")
print(f"  Perturbations MAE: {m2_pert_mae:.6f}")
print(f"  Difference:        {m2_pert_mae - m2_test_mae:.6f} ({((m2_pert_mae/m2_test_mae - 1)*100):.2f}%)")

# 5. Visualize predictions on test set
print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

# Re-adapt for test set if adabatch is enabled (predictions mode)
if adabatch:
    print("\nRe-adapting batch normalization for test set predictions...")
    adapt_batch_norm(marker1_model, x_test, batch_size=32)
    adapt_batch_norm(marker2_model, x_test, batch_size=32)

# Get predictions
m1_test_pred = marker1_model.predict(x_test).flatten()
m2_test_pred = marker2_model.predict(x_test).flatten()

# Combine predictions into (N, 2) array for distribution plots
Y_pred = np.stack([m1_test_pred, m2_test_pred], axis=1)
Y_true = y_test

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Marker 1 - Scatter plot
axes[0, 0].scatter(y_test[:, 0], m1_test_pred, alpha=0.5)
axes[0, 0].plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                [y_test[:, 0].min(), y_test[:, 0].max()], 
                'r--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('True Marker 1 Values')
axes[0, 0].set_ylabel('Predicted Marker 1 Values')
axes[0, 0].set_title(f'Marker 1 Predictions (Test MAE: {m1_test_mae:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Marker 1 - Residuals
residuals_m1 = y_test[:, 0] - m1_test_pred
axes[0, 1].scatter(m1_test_pred, residuals_m1, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Marker 1 Values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Marker 1 Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# Marker 2 - Scatter plot
axes[1, 0].scatter(y_test[:, 1], m2_test_pred, alpha=0.5)
axes[1, 0].plot([y_test[:, 1].min(), y_test[:, 1].max()], 
                [y_test[:, 1].min(), y_test[:, 1].max()], 
                'r--', lw=2, label='Perfect prediction')
axes[1, 0].set_xlabel('True Marker 2 Values')
axes[1, 0].set_ylabel('Predicted Marker 2 Values')
axes[1, 0].set_title(f'Marker 2 Predictions (Test MAE: {m2_test_mae:.4f})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Marker 2 - Residuals
residuals_m2 = y_test[:, 1] - m2_test_pred
axes[1, 1].scatter(m2_test_pred, residuals_m2, alpha=0.5)
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Marker 2 Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Marker 2 Residual Plot')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cellcycle_markers_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to 'cellcycle_markers_results.png'")
plt.show()

# 6. Distribution plots for test set predictions
print("\n" + "="*60)
print("Generating distribution plots...")
print("="*60)

def plot_pred_distributions_01(Y_pred, Y_true=None, bins=60, title_prefix=""):
    """
    Plot distribution histograms and joint density for cell cycle markers.
    
    Y_pred: (N,2) in [0,1]   -> [Cdt1, Geminin]
    Y_true: optional (N,2)   -> overlay as outline
    """
    # clamp just in case
    P = np.clip(Y_pred.astype(float), 0.0, 1.0)
    T = np.clip(Y_true.astype(float), 0.0, 1.0) if Y_true is not None else None

    # 1) Cdt1 (Marker 1) histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(P[:,0], bins=bins, range=(0,1), density=True, alpha=0.7, label="pred", color='skyblue')
    if T is not None:
        ax.hist(T[:,0], bins=bins, range=(0,1), density=True, histtype="step", linewidth=1.5, label="true", color='darkblue')
    ax.set(xlabel="Cdt1 intensity (0–1)", ylabel="Density", title=f"{title_prefix}Cdt1 Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cellcycle_cdt1_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2) Geminin (Marker 2) histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(P[:,1], bins=bins, range=(0,1), density=True, alpha=0.7, label="pred", color='lightcoral')
    if T is not None:
        ax.hist(T[:,1], bins=bins, range=(0,1), density=True, histtype="step", linewidth=1.5, label="true", color='darkred')
    ax.set(xlabel="Geminin intensity (0–1)", ylabel="Density", title=f"{title_prefix}Geminin Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cellcycle_geminin_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3) Joint distribution (2D hexbin)
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(P[:,0], P[:,1], gridsize=50, extent=(0,1,0,1), bins="log", cmap='viridis')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log(count)")
    ax.set(xlabel="Cdt1 (0–1)", ylabel="Geminin (0–1)", title=f"{title_prefix}Joint Density (Predicted)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cellcycle_joint_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4) Joint distribution for true values (if available)
    if T is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        hb = ax.hexbin(T[:,0], T[:,1], gridsize=50, extent=(0,1,0,1), bins="log", cmap='plasma')
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log(count)")
        ax.set(xlabel="Cdt1 (0–1)", ylabel="Geminin (0–1)", title=f"{title_prefix}Joint Density (True)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cellcycle_joint_distribution_true.png', dpi=300, bbox_inches='tight')
        plt.show()

# Generate distribution plots
plot_pred_distributions_01(Y_pred, Y_true=Y_true, bins=60, title_prefix="Test Set - ")
print("\nDistribution plots saved:")
print("  - cellcycle_cdt1_distribution.png")
print("  - cellcycle_geminin_distribution.png")
print("  - cellcycle_joint_distribution.png")
print("  - cellcycle_joint_distribution_true.png")

# 7. Visualize predictions on perturbations set
print("\n" + "="*60)
print("Generating perturbations visualizations...")
print("="*60)

# Re-adapt for perturbations if adabatch is enabled
if adabatch:
    print("\nRe-adapting batch normalization for perturbations predictions...")
    adapt_batch_norm(marker1_model, x_pert, batch_size=32)
    adapt_batch_norm(marker2_model, x_pert, batch_size=32)

# Get predictions for perturbations
m1_pert_pred = marker1_model.predict(x_pert).flatten()
m2_pert_pred = marker2_model.predict(x_pert).flatten()

# Combine predictions
Y_pred_pert = np.stack([m1_pert_pred, m2_pert_pred], axis=1)
Y_true_pert = y_pert

# Create scatter and residual plots for perturbations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Marker 1 - Scatter plot
axes[0, 0].scatter(y_pert[:, 0], m1_pert_pred, alpha=0.5, color='orange')
axes[0, 0].plot([y_pert[:, 0].min(), y_pert[:, 0].max()], 
                [y_pert[:, 0].min(), y_pert[:, 0].max()], 
                'r--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('True Marker 1 Values')
axes[0, 0].set_ylabel('Predicted Marker 1 Values')
axes[0, 0].set_title(f'Marker 1 Predictions - Perturbations (MAE: {m1_pert_mae:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Marker 1 - Residuals
residuals_m1_pert = y_pert[:, 0] - m1_pert_pred
axes[0, 1].scatter(m1_pert_pred, residuals_m1_pert, alpha=0.5, color='orange')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Marker 1 Values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Marker 1 Residual Plot - Perturbations')
axes[0, 1].grid(True, alpha=0.3)

# Marker 2 - Scatter plot
axes[1, 0].scatter(y_pert[:, 1], m2_pert_pred, alpha=0.5, color='green')
axes[1, 0].plot([y_pert[:, 1].min(), y_pert[:, 1].max()], 
                [y_pert[:, 1].min(), y_pert[:, 1].max()], 
                'r--', lw=2, label='Perfect prediction')
axes[1, 0].set_xlabel('True Marker 2 Values')
axes[1, 0].set_ylabel('Predicted Marker 2 Values')
axes[1, 0].set_title(f'Marker 2 Predictions - Perturbations (MAE: {m2_pert_mae:.4f})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Marker 2 - Residuals
residuals_m2_pert = y_pert[:, 1] - m2_pert_pred
axes[1, 1].scatter(m2_pert_pred, residuals_m2_pert, alpha=0.5, color='green')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Marker 2 Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Marker 2 Residual Plot - Perturbations')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cellcycle_markers_perturbations_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to 'cellcycle_markers_perturbations_results.png'")
plt.show()

# # Distribution plots for perturbations
# plot_pred_distributions_01(Y_pred_pert, Y_true=Y_true_pert, bins=60, title_prefix="Perturbations - ")
# print("\nPerturbations distribution plots saved:")
# print("  - cellcycle_cdt1_distribution.png (updated with perturbations)")
# print("  - cellcycle_geminin_distribution.png (updated with perturbations)")
# print("  - cellcycle_joint_distribution.png (updated with perturbations)")
# print("  - cellcycle_joint_distribution_true.png (updated with perturbations)")

# 8. Comparative metrics visualization
print("\n" + "="*60)
print("Generating comparative metrics plots...")
print("="*60)

# Create bar plot comparing Test vs Perturbations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Marker 1 comparison
datasets = ['Test', 'Perturbations']
m1_maes = [m1_test_mae, m1_pert_mae]
bars1 = axes[0].bar(datasets, m1_maes, color=['skyblue', 'orange'], edgecolor='black')
axes[0].set_ylabel('MAE')
axes[0].set_title('Marker 1 (Cdt1) - MAE Comparison')
axes[0].grid(True, alpha=0.3, axis='y')
for bar in bars1:
    height = bar.get_height()
    axes[0].annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

# Marker 2 comparison
m2_maes = [m2_test_mae, m2_pert_mae]
bars2 = axes[1].bar(datasets, m2_maes, color=['lightcoral', 'green'], edgecolor='black')
axes[1].set_ylabel('MAE')
axes[1].set_title('Marker 2 (Geminin) - MAE Comparison')
axes[1].grid(True, alpha=0.3, axis='y')
for bar in bars2:
    height = bar.get_height()
    axes[1].annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

plt.tight_layout()
plt.savefig('cellcycle_test_vs_perturbations_comparison.png', dpi=300, bbox_inches='tight')
print("\nComparative plot saved to 'cellcycle_test_vs_perturbations_comparison.png'")
plt.show()

print("\n" + "="*60)
print("Training and evaluation complete!")
print("="*60)
