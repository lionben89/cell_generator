import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from metrics import tf_pearson_corr

###############--SETTINGS--##########################
num_samples = 100  # Number of samples to visualize
dataset_to_visualize = "perturbations"  # Options: "train", "val", "test"
plot_examples = False  # Whether to generate visualization plots
create_all_predictions = True  # Whether to save all mask predictions as .npy files

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
    # Load dataset
    base_data_dir = "/groups/assafza_group/assafza/Gad/Cell_Cycle_Data"
    
    print(f"\nLoading {dataset_to_visualize} set...")
    x_data, y_data = load_cell_cycle_data(os.path.join(base_data_dir, dataset_to_visualize))
    
    # Normalize images
    print("\nNormalizing images...")
    x_data = x_data.astype("float32")
    
    # Compute mean and std for normalization
    mean = np.mean(x_data)
    std = np.std(x_data)
    print(f"{dataset_to_visualize.capitalize()} set - Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Apply normalization
    x_data_norm = (x_data - mean) / std
    x_data_orig = x_data  # Keep original for display
    
    ###############--LOAD MODELS--##########################
    
    print("\nLoading trained models...")
    
    # Load regressors
    marker1_model = tf.keras.models.load_model('cellcycle_marker1.h5')
    marker2_model = tf.keras.models.load_model('cellcycle_marker2.h5')
    
    # Load mask interpreters
    from mi_reg_cellcycle import MaskInterpreterRegression
    from models.UNETO import get_unet
    
    # Create adaptors
    adaptor_m1 = get_unet((64, 64, 2), activation="sigmoid")
    adaptor_m2 = get_unet((64, 64, 2), activation="sigmoid")
    
    # Create mask interpreters
    mi_marker1 = MaskInterpreterRegression(
        patch_size=(64, 64, 1),
        adaptor=adaptor_m1,
        regressor=marker1_model,
        pcc_target=0.95
    )
    
    mi_marker2 = MaskInterpreterRegression(
        patch_size=(64, 64, 1),
        adaptor=adaptor_m2,
        regressor=marker2_model,
        pcc_target=0.95
    )
    
    # Compile (required before loading weights)
    mi_marker1.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        similarity_loss_weight=1.0,
        mask_loss_weight=1.0,
        noise_scale=0.5,
        target_loss_weight=1.75
    )
    
    mi_marker2.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        similarity_loss_weight=1.0,
        mask_loss_weight=1.0,
        noise_scale=0.5,
        target_loss_weight=1.75
    )
    
    # Build models
    mi_marker1(np.random.random((1, 64, 64, 1)).astype(np.float32))
    mi_marker2(np.random.random((1, 64, 64, 1)).astype(np.float32))
    
    # Load weights using the full model loading approach
    print("Loading Mask Interpreter weights...")
    
    # Load Marker 1 mask interpreter
    mi_marker1_pt = keras.models.load_model("./cellcycle_mi_marker1")
    mi_marker1.set_weights(mi_marker1_pt.get_weights())
    print("✓ Loaded Marker 1 mask interpreter weights")
    
    # Load Marker 2 mask interpreter
    mi_marker2_pt = keras.models.load_model("./cellcycle_mi_marker2")
    mi_marker2.set_weights(mi_marker2_pt.get_weights())
    print("✓ Loaded Marker 2 mask interpreter weights")


    
    ###############--VISUALIZATION--##########################
    
    if plot_examples:
    
        if plot_examples:
            print("\n" + "="*60)
            print("Generating visualizations...")
            print("="*60)
        
        # Create output directories
        output_dir_m1 = f"./cellcycle_mi_images_marker1_{dataset_to_visualize}"
        output_dir_m2 = f"./cellcycle_mi_images_marker2_{dataset_to_visualize}"
        
        try:
            os.makedirs(output_dir_m1, exist_ok=True)
            os.makedirs(output_dir_m2, exist_ok=True)
        except Exception as e:
            print(f"Error creating directories: {e}")
        
        sample_indices = range(min(num_samples, x_data.shape[0]))
        
        for idx in tqdm(sample_indices, desc="Visualizing samples"):
            # Get normalized image and its original version for display
            img_norm = x_data_norm[idx]  # normalized image (64, 64, 1)
            img_orig = x_data_orig[idx]  # un-normalized image for plotting
            true_marker1 = y_data[idx, 0]
            true_marker2 = y_data[idx, 1]
            
            ###############--MARKER 1--##########################
            
            # 1. Regressor Prediction on Original Image
            pred_marker1_orig = marker1_model.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0, 0]
            
            # 2. Generate Importance Mask Using the MaskInterpreter
            mask_m1 = mi_marker1(np.expand_dims(img_norm, axis=0))
            mask_m1 = np.squeeze(mask_m1)  # shape becomes (64, 64) or (64, 64, 1)
            
            # Ensure mask is 2D for visualization
            if mask_m1.ndim == 3:
                mask_m1_vis = mask_m1[:, :, 0]
            else:
                mask_m1_vis = mask_m1
                
            # Expand for broadcasting
            if mask_m1.ndim == 2:
                mask_m1_expanded = np.expand_dims(mask_m1, axis=-1)
            else:
                mask_m1_expanded = mask_m1
            
            # 3. Generate a Noisy (Adapted) Image Using the Importance Mask
            noise_m1 = tf.random.normal(shape=img_norm.shape, 
                                     stddev=mi_marker1.noise_scale, 
                                     dtype=tf.float64).numpy()
        
        adapted_image_m1 = (mask_m1_expanded * img_norm.astype(np.float64)) + \
                           ((1 - mask_m1_expanded) * noise_m1)
        
        # 4. Regressor Prediction on Adapted Image
        pred_marker1_adapted = marker1_model.predict(
            np.expand_dims(adapted_image_m1.astype(np.float32), axis=0), verbose=0
        )[0, 0]
        
        # 5. Revert Normalization for Display
        adapted_image_m1_orig = (adapted_image_m1 * std) + mean
        adapted_image_m1_orig = np.clip(adapted_image_m1_orig, 0, 1)
        
        # Calculate mask efficacy (correlation between predictions)
        mask_efficacy_m1 = np.corrcoef([pred_marker1_orig], [pred_marker1_adapted])[0, 1]
        
        # 6. Plot the Results for Marker 1
        plt.figure(figsize=(15, 4))
        
        # Subplot 1: Original image
        plt.subplot(1, 4, 1)
        plt.imshow(img_orig[:, :, 0], cmap='gray')
        plt.title(f"Original\nTrue Marker1: {true_marker1:.3f}\nPred: {pred_marker1_orig:.3f}")
        plt.axis('off')
        
        # Subplot 2: Adapted (noisy) image
        plt.subplot(1, 4, 2)
        plt.imshow(adapted_image_m1_orig[:, :, 0], cmap='gray')
        plt.title(f"Adapted (Noisy)\nPred: {pred_marker1_adapted:.3f}")
        plt.axis('off')
        
        # Subplot 3: Importance mask
        plt.subplot(1, 4, 3)
        plt.imshow(mask_m1_vis, cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Importance Mask")
        plt.axis('off')
        
        # Subplot 4: Original image with mask overlay
        plt.subplot(1, 4, 4)
        plt.imshow(img_orig[:, :, 0], cmap='gray')
        plt.imshow(mask_m1_vis, cmap='jet', alpha=0.5)
        plt.title(f"Mask Overlay\nEfficacy: {mask_efficacy_m1:.3f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir_m1}/{idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        ###############--MARKER 2--##########################
        
        # 1. Regressor Prediction on Original Image
        pred_marker2_orig = marker2_model.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0, 0]
        
        # 2. Generate Importance Mask Using the MaskInterpreter
        mask_m2 = mi_marker2(np.expand_dims(img_norm, axis=0))
        mask_m2 = np.squeeze(mask_m2)
        
        # Ensure mask is 2D for visualization
        if mask_m2.ndim == 3:
            mask_m2_vis = mask_m2[:, :, 0]
        else:
            mask_m2_vis = mask_m2
            
        # Expand for broadcasting
        if mask_m2.ndim == 2:
            mask_m2_expanded = np.expand_dims(mask_m2, axis=-1)
        else:
            mask_m2_expanded = mask_m2
        
        # 3. Generate a Noisy (Adapted) Image
        noise_m2 = tf.random.normal(shape=img_norm.shape, 
                                     stddev=mi_marker2.noise_scale, 
                                     dtype=tf.float64).numpy()
        
        adapted_image_m2 = (mask_m2_expanded * img_norm.astype(np.float64)) + \
                           ((1 - mask_m2_expanded) * noise_m2)
        
        # 4. Regressor Prediction on Adapted Image
        pred_marker2_adapted = marker2_model.predict(
            np.expand_dims(adapted_image_m2.astype(np.float32), axis=0), verbose=0
        )[0, 0]
        
        # 5. Revert Normalization for Display
        adapted_image_m2_orig = (adapted_image_m2 * std) + mean
        adapted_image_m2_orig = np.clip(adapted_image_m2_orig, 0, 1)
        
        # Calculate mask efficacy
        mask_efficacy_m2 = np.corrcoef([pred_marker2_orig], [pred_marker2_adapted])[0, 1]
        
        # 6. Plot the Results for Marker 2
        plt.figure(figsize=(15, 4))
        
        # Subplot 1: Original image
        plt.subplot(1, 4, 1)
        plt.imshow(img_orig[:, :, 0], cmap='gray')
        plt.title(f"Original\nTrue Marker2: {true_marker2:.3f}\nPred: {pred_marker2_orig:.3f}")
        plt.axis('off')
        
        # Subplot 2: Adapted (noisy) image
        plt.subplot(1, 4, 2)
        plt.imshow(adapted_image_m2_orig[:, :, 0], cmap='gray')
        plt.title(f"Adapted (Noisy)\nPred: {pred_marker2_adapted:.3f}")
        plt.axis('off')
        
        # Subplot 3: Importance mask
        plt.subplot(1, 4, 3)
        plt.imshow(mask_m2_vis, cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Importance Mask")
        plt.axis('off')
        
        # Subplot 4: Original image with mask overlay
        plt.subplot(1, 4, 4)
        plt.imshow(img_orig[:, :, 0], cmap='gray')
        plt.imshow(mask_m2_vis, cmap='jet', alpha=0.5)
        plt.title(f"Mask Overlay\nEfficacy: {mask_efficacy_m2:.3f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir_m2}/{idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
        print(f"\nVisualization complete!")
        print(f"Marker 1 images saved to: {output_dir_m1}")
        print(f"Marker 2 images saved to: {output_dir_m2}")
        print("="*60)
    
    ###############--SAVE ALL PREDICTIONS--##########################
    
    if create_all_predictions:
        print("\n" + "="*60)
        print("Saving all mask predictions...")
        print("="*60)
        
        # Create predictions directories next to images and labels
        base_data_dir = "/groups/assafza_group/assafza/Gad/Cell_Cycle_Data"
        predictions_dir_m1 = os.path.join(base_data_dir, dataset_to_visualize, "predictions_marker1")
        predictions_dir_m2 = os.path.join(base_data_dir, dataset_to_visualize, "predictions_marker2")
        
        try:
            os.makedirs(predictions_dir_m1, exist_ok=True)
            os.makedirs(predictions_dir_m2, exist_ok=True)
            print(f"Created directories:")
            print(f"  {predictions_dir_m1}")
            print(f"  {predictions_dir_m2}")
        except Exception as e:
            print(f"Error creating prediction directories: {e}")
            exit(1)
        
        # Get list of all image files to match filenames
        images_dir = os.path.join(base_data_dir, dataset_to_visualize, "images")
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
        
        # Process all samples
        for idx, img_file in enumerate(tqdm(image_files, desc="Saving predictions")):
            filename = os.path.basename(img_file)
            
            # Get normalized image
            img_norm = x_data_norm[idx]
            
            # Generate masks for both markers
            mask_m1 = mi_marker1(np.expand_dims(img_norm, axis=0))
            mask_m1 = np.squeeze(mask_m1)  # Convert to numpy and remove batch dim
            
            mask_m2 = mi_marker2(np.expand_dims(img_norm, axis=0))
            mask_m2 = np.squeeze(mask_m2)
            
            # Save masks with the same filename as the original image
            np.save(os.path.join(predictions_dir_m1, filename), mask_m1)
            np.save(os.path.join(predictions_dir_m2, filename), mask_m2)
        
        print(f"\nPrediction saving complete!")
        print(f"Marker 1 masks saved to: {predictions_dir_m1}")
        print(f"Marker 2 masks saved to: {predictions_dir_m2}")
        print(f"Total predictions saved: {len(image_files)} per marker")
        print("="*60)

