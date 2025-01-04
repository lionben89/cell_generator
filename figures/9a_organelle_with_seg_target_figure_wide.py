import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from cell_imaging_utils.image.image_utils import ImageUtils
import os
import cv2
import pandas as pd
from figure_config import figure_config,get_scalebar

params = [
    {"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","th":0.60,"slice":28},
    # {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","th":0.40,"slice":31},
    # {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","th":0.2,"slice":37},
    # {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","th":0.2,"slice":30},
]

def auto_balance(image):
    """Auto balance the image similar to ImageJ's Auto Contrast function."""
    # Convert to float32 to avoid issues with overflow/underflow
    image = image.astype(np.float32)
    
    # Calculate the 0.1th and 99.9th percentiles
    plow, phigh = np.percentile(image, (0.1, 99.9))
    
    # Stretch the values to the full range [0, 1]
    image = np.clip((image - plow) / (phigh - plow), 0, 1)
    
    return image

def collect_images(param, image_index):
    # Construct paths for the required images
    base_path = "{}/predictions_agg/{}".format(param["model"], image_index)
    file_paths = {
        "prediction": "{}/unet_prediction_{}.tiff".format(base_path, image_index),
        "input": "{}/input_{}.tiff".format(base_path, image_index),
        "mask": "{}/{}/mask_{}.tiff".format(base_path, '{0:.2f}'.format(param["th"]), image_index),
        "noisy_prediction": "{}/{}/noisy_unet_prediction_{}.tiff".format(base_path, '{0:.2f}'.format(param["th"]), image_index),
        "ground_truth": "{}/target_{}.tiff".format(base_path, image_index),
        "seg_ground_truth": "{}/seg_target_{}.tiff".format(base_path, image_index)
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

    best_slice_index = param["slice"]

    # Extract the most informative slice from each stack using the index found from the prediction stack
    most_informative_slices = {key: auto_balance(stack[best_slice_index, :, :, 0]) for key, stack in stacks.items()}

    return most_informative_slices["input"], most_informative_slices["mask"], most_informative_slices["prediction"], most_informative_slices["noisy_prediction"], most_informative_slices["ground_truth"], most_informative_slices["seg_ground_truth"], best_slice_index

def overlay_images(input_image, mask_image, seg_gt_image):
    # Convert the input image to RGB if it's not already
    input_image_rgb = cv2.cvtColor((input_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Ensure mask_image and seg_gt_image are single channel
    if len(mask_image.shape) == 3:
        mask_image = mask_image[:, :, 0]
    if len(seg_gt_image.shape) == 3:
        seg_gt_image = seg_gt_image[:, :, 0]

    # Create a color overlay for the mask and segmentation ground truth
    turquoise = np.array([64, 224, 208]) / 255.0  # Turquoise color
    purple = np.array([128, 0, 128]) / 255.0      # Purple color

    # Create an overlay image
    overlay = input_image_rgb.copy().astype(np.float32)

    # Apply mask color
    mask_indices = mask_image > 0
    overlay[mask_indices] = 0.5 * turquoise * 255 + 0.5 * overlay[mask_indices]

    # Apply segmentation ground truth color
    seg_gt_indices = seg_gt_image > 0
    overlay[seg_gt_indices] = 0.5 * purple * 255 + 0.5 * overlay[seg_gt_indices]

    # Combine both where both mask and segmentation ground truth are present
    combined_indices = mask_indices & seg_gt_indices
    overlay[combined_indices] = 0.5 * (turquoise * 255 + purple * 255) + 0.5 * overlay[combined_indices]

    return overlay.astype(np.uint8)

# Create plot for a single organelle
def plot_organelle(image_index, param, save_path):
    # Create a figure with a tight layout
    fig = plt.figure(figsize=(10, 4.0))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[2, 1, 1])
    gs.update(wspace=0.01, hspace=0.01)  # Minimal space between images
    
    input_image, mask_image, prediction_original, prediction_noisy, ground_truth, seg_ground_truth, slice_index = collect_images(param, image_index)
    mask_seg_gt_overlay = overlay_images(input_image, mask_image, seg_ground_truth)
    
    # Add a title with the organelle name, closer to the images
    fig.text(0.5,0.98,param["organelle"], fontsize=figure_config["organelle"],fontname=figure_config["font"], color='black', ha='center', va='top')
    
    #Add subtitle
    mean_pcc = pd.read_csv("{}/predictions_agg/pcc_resuls.csv".format(param["model"]))["{0:.1f}".format(param["th"])].mean()
    mask_size = 1- (pd.read_csv("{}/predictions_agg/mask_size_resuls.csv".format(param["model"]))["{0:.1f}".format(param["th"])].mean())
    fig.text(0.5,0.92,"TH={:.1f}, Mask efficacy={:.2f} [PCC], Mean noise vol.={:.2f}%".format(param["th"],mean_pcc,mask_size*100),  fontsize=figure_config["subtitle"],fontname=figure_config["font"], color='black', ha='center', va='top')
    
    # Adjust subplot parameters to bring title closer to images
    plt.subplots_adjust(top=0.87)

    # Add the first image at the top (100% width, 50% height)
    ax1 = plt.subplot(gs[:, 0])
    ax1.imshow(mask_seg_gt_overlay)
    ax1.axis('off')  # Hide axes
    ax1.text(0.01, 0.975, "Input Image + Mask overlay + GT segmentation", transform=ax1.transAxes,  fontsize=figure_config["text"],fontname=figure_config["font"], fontweight="bold",color='white', ha='left', va='top')
    ax1.add_artist(get_scalebar())
    
    # Add the second image at the bottom left (50% width, 25% height)
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(input_image, cmap='gray')
    ax2.axis('off')
    ax2.text(0.01, 0.95, "Input Image", transform=ax2.transAxes,  fontsize=figure_config["text"],fontname=figure_config["font"], fontweight="bold",color='white', ha='left', va='top')
    ax2.add_artist(get_scalebar())
    
    # Add the third image at the bottom right (50% width, 25% height)
    ax3 = plt.subplot(gs[0, 2])
    ax3.imshow(ground_truth, cmap='gray')
    ax3.axis('off')
    ax3.text(0.01, 0.95, "Ground Truth", transform=ax3.transAxes,  fontsize=figure_config["text"],fontname=figure_config["font"],fontweight="bold", color='white', ha='left', va='top')

    # Add the fourth image at the very bottom left (50% width, 25% height)
    ax4 = plt.subplot(gs[1, 1])
    ax4.imshow(prediction_noisy, cmap='gray')
    ax4.axis('off')
    ax4.text(0.01, 0.95, "Noisy Prediction", transform=ax4.transAxes,  fontsize=figure_config["text"],fontname=figure_config["font"],fontweight="bold", color='white', ha='left', va='top')

    # Add the fifth image at the very bottom right (50% width, 25% height)
    ax5 = plt.subplot(gs[1, 2])
    ax5.imshow(prediction_original, cmap='gray')
    ax5.axis('off')
    ax5.text(0.01, 0.95, "Prediction", transform=ax5.transAxes,  fontsize=figure_config["text"],fontname=figure_config["font"],fontweight="bold", color='white', ha='left', va='top')

    plt.savefig(save_path, bbox_inches='tight',pad_inches=0.1)
    plt.close(fig)

# Example usage
image_index = 4

# Directory to save individual plots
output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)
organelle_names = []

# Generate and save individual organelle plots
for param in params:
    print(param["organelle"])
    organelle_names.append(param["organelle"])
    plot_organelle(image_index, param=param, save_path=os.path.join(output_dir, 'validation_{}_wide.png'.format(param["organelle"])))
