import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from scipy.stats import entropy
from cell_imaging_utils.image.image_utils import ImageUtils
import os
import cv2
from figure_config import figure_config, get_scalebar

params = [
          {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_noise_1.5_sim_0.0_target_2.0_mask_1.0_mse","th":0.2,"slices":[44], "title":"S=0 T=6 M=1"},
          {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_noise_1.5_sim_1.0_target_2.0_mask_1.0_mse","th":0.2,"slices":[44], "title":"S=1 T=6 M=1"},
          {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_noise_1.5_sim_1.0_target_0.0_mask_4.0_mse","th":0.2,"slices":[44], "title":"S=1 T=0 M=4"},
          {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_noise_1.5_sim_1.0_target_6.0_mask_4.0_mse","th":0.2,"slices":[44], "title":"S=1 T=6 M=4"},
          ]

def auto_balance(image):
    """Auto balance the image similar to ImageJ's Auto Contrast function."""
    # Convert to float32 to avoid issues with overflow/underflow
    image = image.astype(np.float32)
    
    # Calculate the 2nd and 98th percentiles
    plow, phigh = np.percentile(image, (0.1, 99.9))
    
    # Stretch the values to the full range [0, 255]
    # image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = np.clip((image - plow) / (phigh - plow), 0, 1)
    
    return image

# Function to overlay mask on input image with a turquoise color
def overlay_mask(input_image, mask_image):
    # Convert the input image to RGB if it's not already
    if len(input_image.shape) == 2 or input_image.shape[2] != 3:
        input_image = cv2.cvtColor(input_image*255.0, cv2.COLOR_BGR2RGB)

    # Ensure mask_image and seg_gt_image are single channel
    if len(mask_image.shape) == 3:
        mask_image = mask_image[:, :, 0]

    # Create a color overlay for the mask and segmentation ground truth
    turquoise = np.array([64, 224, 208]) / 255.0  # Turquoise color

    # Create an overlay image
    overlay = input_image.copy().astype(np.float32)

    # Apply mask color
    mask_indices = mask_image > 0
    overlay[mask_indices] = 0.5 * turquoise * 255 + 0.5 * overlay[mask_indices]

    return overlay.astype(np.uint8)

def calculate_entropy(slice):
    """Calculate Shannon entropy of a given image slice."""
    hist, _ = np.histogram(slice, bins=256, range=(0, 1))
    prob_dist = hist / hist.sum()
    return entropy(prob_dist, base=2)

def collect_images(param, image_index):
    # Construct paths for the required images
    base_path = "{}/predictions_agg/{}".format(param["model"], image_index)
    file_paths = {
        "prediction": "{}/unet_prediction_{}.tiff".format(base_path, image_index),
        "input": "{}/input_{}.tiff".format(base_path, image_index),
        "mask": "{}/{}/mask_{}.tiff".format(base_path, '{0:.2f}'.format(param["th"]), image_index),
        "noisy_prediction": "{}/{}/noisy_unet_prediction_{}.tiff".format(base_path, '{0:.2f}'.format(param["th"]), image_index),
        "ground_truth": "{}/target_{}.tiff".format(base_path, image_index)
    }

    # Check if files exist
    for key, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")

    # Load and normalize the stacks
    stacks = {}
    for key, path in file_paths.items():
        stack = ImageUtils.image_to_ndarray(ImageUtils.imread(path))
        stack = ImageUtils.normalize(stack, max_value=1.0, dtype=np.float32)
        stacks[key] = stack

    best_slice_index = param["slices"][image_index]

    # Extract the most informative slice from each stack using the index found from the prediction stack
    most_informative_slices = {key: auto_balance(stack[best_slice_index, :, :, 0]) for key, stack in stacks.items()}

    return most_informative_slices["input"], most_informative_slices["mask"], most_informative_slices["prediction"], most_informative_slices["noisy_prediction"], most_informative_slices["ground_truth"],best_slice_index

# Resize images to fit the layout
def resize_image(image, size=(int(312*1.5), int(462*1.5))):
    return resize(image, size, anti_aliasing=True)

# Create plot comparing multiple params
def plot_params_comparison(params, image_index, save_path):
    
    """
    Create a plot with:
    - Top 2 rows: input+mask and prediction noisy for each param (len(params) columns)
    - Bottom 1 row: prediction original and ground truth from first param only (2 big images)
    """
    from matplotlib.gridspec import GridSpec
    
    num_params = len(params)
    # Make figure square: 4 params wide, each top cell is unit 1, bottom cells are 4 units tall
    # Total height: 2 (top rows) + 4 (bottom row) = 6 units, width: 4 units, so 4x1.5=6 for square
    fig = plt.figure(figsize=(num_params * 2.0, num_params * 1.61))
    
    # Add figure title using organelle from first param
    fig.suptitle(params[0]["organelle"], fontsize=18, fontname=figure_config["font"], weight='bold')
    
    # Create GridSpec with height ratios: top 2 rows get 1 unit each, bottom row gets 4 units
    gs = GridSpec(3, max(num_params, 2), figure=fig, hspace=0.02, wspace=0.02,
                  height_ratios=[1, 1, 2])
    
    # Top section: input+mask and prediction noisy for each param
    for i, param in enumerate(params):
        input_image, mask_image, prediction_original, prediction_noisy, ground_truth, slice_index = collect_images(param, image_index)
        
        # Row 0: Input+Mask overlay
        ax0 = fig.add_subplot(gs[0, i])
        overlay_image = overlay_mask(input_image, mask_image)
        ax0.imshow(overlay_image)
        if i == 0:
            ax0.add_artist(get_scalebar())
            ax0.text(0.02, 0.98, 'Bright-field+Mask', transform=ax0.transAxes,
                    fontsize=8, fontname=figure_config["font"],
                    color='white', weight='bold', va='top', ha='left')
        # Set column title as model name
        title = ax0.set_title(param["title"].split("/")[-1], fontsize=12, pad=4)
        title.set_position([0.5, 1.05])
        ax0.get_xaxis().set_visible(False)
        ax0.get_yaxis().set_visible(False)
        
        # Row 1: Prediction noisy
        ax1 = fig.add_subplot(gs[1, i])
        ax1.imshow(prediction_noisy, cmap='gray')
        if i == 0:
            ax1.text(0.02, 0.98, 'Noisy prediction', transform=ax1.transAxes,
                    fontsize=8, fontname=figure_config["font"],
                    color='white', weight='bold', va='top', ha='left')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
    
    # Bottom section: prediction original and ground truth from first param only
    input_image, mask_image, prediction_original, prediction_noisy, ground_truth, slice_index = collect_images(params[0], image_index)
    
    # Prediction original (left half of bottom row)
    ax2 = fig.add_subplot(gs[2, :num_params//2])
    ax2.imshow(prediction_original, cmap='gray')
    ax2.add_artist(get_scalebar())
    ax2.text(0.02, 0.98, 'Prediction', transform=ax2.transAxes,
            fontsize=8, fontname=figure_config["font"],
            color='white', weight='bold', va='top', ha='left')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    # Ground truth (right half of bottom row)
    ax3 = fig.add_subplot(gs[2, num_params//2:])
    ax3.imshow(ground_truth, cmap='gray')
    ax3.text(0.02, 0.98, 'Ground Truth', transform=ax3.transAxes,
            fontsize=8, fontname=figure_config["font"],
            color='white', weight='bold', va='top', ha='left')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.02, wspace=0.02)
    plt.close(fig)

# Directory to save individual plots
output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)

# Generate plot comparing all params
plot_params_comparison(params, image_index=0, save_path=os.path.join(output_dir, 'params_comparison.png'))