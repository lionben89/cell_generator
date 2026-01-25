import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from cell_imaging_utils.image.image_utils import ImageUtils
import os
import cv2
import pandas as pd
from figure_config import figure_config,scalebar,get_scalebar

params = [
    {"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","noise_vol":0.9,"slice":26},
    {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","noise_vol":0.3,"slice":29},
    {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","noise_vol":0.3,"slice":36},
    {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","noise_vol":0.8,"slice":22},
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

def collect_images(param,image_index, interpreter):
    # Construct paths for the required images
    base_path = "{}/compare/{}/{}".format(param["model"],interpreter, image_index)
    file_paths = {
        "prediction": "{}/prediction_{}.tiff".format(base_path, image_index),
        "input": "{}/input_{}.tiff".format(base_path, image_index),
        interpreter: "{}/mask_{}_noisevol_{}.tiff".format(base_path, image_index, param["noise_vol"]),
        "noisy_prediction": "{}/noisy_prediction_{}_noisevol_{}.tiff".format(base_path, image_index,param["noise_vol"]),
        "ground_truth": "{}/target_{}.tiff".format(base_path, image_index),
        "seg_ground_truth": "{}/target_seg_{}.tiff".format(base_path, image_index)
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

    return most_informative_slices["input"], most_informative_slices[interpreter], most_informative_slices["prediction"], most_informative_slices["noisy_prediction"], most_informative_slices["ground_truth"], most_informative_slices["seg_ground_truth"], best_slice_index

def overlay_images(input_image, mask_image):
    # Convert the input image to RGB if it's not already
    input_image_rgb = cv2.cvtColor((input_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Create a color overlay for the mask and segmentation ground truth
    turquoise = np.array([64, 224, 208]) / 255.0  # Turquoise color
    purple = np.array([128, 0, 128]) / 255.0      # Purple color

    # Create an overlay image
    overlay = input_image_rgb.copy().astype(np.float32)

    # Apply mask color
    mask_indices = mask_image > 0
    overlay[mask_indices] = 0.5 * turquoise * 255 + 0.5 * overlay[mask_indices]

    return overlay.astype(np.uint8)

# Create plot for a single organelle
def plot_organelle(image_index, param, save_path):
    # Create a figure with a tight layout
    fig = plt.figure(figsize=(6,4.5))
    gs = gridspec.GridSpec(3,6,height_ratios=[1,1,1.5])
    gs.update(wspace=0.01, hspace=0.01)  # Minimal space between images
    
    input_image, mask_interpreter_image, prediction_original, mask_interpreter_prediction_noisy, ground_truth, seg_ground_truth, slice_index = collect_images(param, image_index, interpreter="mask_interperter")
    _, saliency_image, _, saliency_prediction_noisy, _, _, _ = collect_images(param, image_index, interpreter="saliency")
    _, gradcam_image, _, gradcam_prediction_noisy, _, _, _ = collect_images(param, image_index, interpreter="gradcam")
    mask_interpreter_overlay = overlay_images(input_image, mask_interpreter_image)
    saliency_overlay = overlay_images(input_image, saliency_image)
    gradcam_overlay = overlay_images(input_image, gradcam_image)
    
    # Add a title with the organelle name, closer to the images
    fig.text(0.5,0.98,param["organelle"], fontsize=figure_config["organelle"],fontname=figure_config["font"], color='black', ha='center', va='top')
    
    #Add subtitle
    mask_interpreter_pcc = pd.read_csv("{}/mask_interperter_evaluation_results_pearson.csv".format(param["model"]))["{0:.1f}".format(param["noise_vol"])].values[image_index]
    saliency_pcc = pd.read_csv("{}/saliency_evaluation_results_pearson.csv".format(param["model"]))["{0:.1f}".format(param["noise_vol"])].values[image_index]
    gradcam_pcc = pd.read_csv("{}/gradcam_evaluation_results_pearson.csv".format(param["model"]))["{0:.1f}".format(param["noise_vol"])].values[image_index]
    mask_size = param["noise_vol"]
    fig.text(0.5,0.94, "Noise vol.={:.2f}%".format(mask_size*100), fontsize=figure_config["subtitle"],fontname=figure_config["font"], color='black', ha='center', va='top')
    
    # Adjust subplot parameters to bring title closer to images
    plt.subplots_adjust(top=0.9)

    # Add the first image at the top left (33% width, 25% height)
    ax1 = plt.subplot(gs[0, :2])
    ax1.imshow(mask_interpreter_overlay)
    ax1.axis('off')  # Hide axes
    ax1.set_adjustable('box')
    ax1.text(0.01, 0.95, "Mask Interpreter", transform=ax1.transAxes, fontweight="bold",fontsize=figure_config["text"],fontname=figure_config["font"], color='white', ha='left', va='top')
    ax1.add_artist(scalebar)
    
    # Add the second image at the top middle (33% width, 25% height)
    ax2 = plt.subplot(gs[0, 2:4])
    ax2.imshow(gradcam_overlay, cmap='gray')
    ax2.axis('off')
    ax2.set_adjustable('box')
    ax2.text(0.01, 0.95, "GRADCAM", transform=ax2.transAxes, fontweight="bold",fontsize=figure_config["text"],fontname=figure_config["font"], color='white', ha='left', va='top')

    # Add the third image at the top right (33% width, 25% height)
    ax3 = plt.subplot(gs[0, 4:])
    ax3.imshow(saliency_overlay, cmap='gray')
    ax3.axis('off')
    ax3.set_adjustable('box')
    ax3.text(0.01, 0.95, "Saliency", transform=ax3.transAxes, fontweight="bold",fontsize=figure_config["text"],fontname=figure_config["font"], color='white', ha='left', va='top')

    # Add the forth image at the middle left (33% width, 25% height)
    ax4 = plt.subplot(gs[1, :2])
    ax4.imshow(mask_interpreter_prediction_noisy, cmap='gray')
    ax4.axis('off')  # Hide axes
    ax4.set_adjustable('box')
    ax4.text(0.01, 0.95, "PCC={0:.2f}".format(mask_interpreter_pcc), transform=ax4.transAxes,fontweight="bold",fontsize=figure_config["text"],fontname=figure_config["font"], color='white', ha='left', va='top')

    # Add the fifth image at the middle middle (33% width, 25% height)
    ax5 = plt.subplot(gs[1, 2:4])
    ax5.imshow(gradcam_prediction_noisy, cmap='gray')
    ax5.axis('off')  # Hide axes
    ax5.set_adjustable('box')
    ax5.text(0.01, 0.95, "PCC={0:.2f}".format(gradcam_pcc), transform=ax5.transAxes, fontweight="bold",fontsize=figure_config["text"],fontname=figure_config["font"], color='white', ha='left', va='top')
    
    # Add the sixth image at the middle right (33% width, 25% height)
    ax6 = plt.subplot(gs[1, 4:])
    ax6.imshow(saliency_prediction_noisy, cmap='gray')
    ax6.axis('off')  # Hide axes
    ax6.set_adjustable('box')
    ax6.text(0.01, 0.95, "PCC={0:.2f}".format(saliency_pcc), transform=ax6.transAxes, fontweight="bold",fontsize=figure_config["text"],fontname=figure_config["font"], color='white', ha='left', va='top')
    
    # Add the seventh image at the bottom right (50% width, 50% height)
    ax7 = plt.subplot(gs[2, :3])
    ax7.imshow(prediction_original, cmap='gray')
    ax7.axis('off')
    ax7.set_adjustable('box')
    ax7.text(0.01, 0.95, "Prediction", transform=ax7.transAxes, fontweight="bold",fontsize=figure_config["text"],fontname=figure_config["font"], color='white', ha='left', va='top')
    ax7.add_artist(get_scalebar())
    
    # Add the eighth image at the bottom right (50% width, 50% height)
    ax8 = plt.subplot(gs[2, 3:])
    ax8.imshow(ground_truth, cmap='gray')
    ax8.axis('off')
    ax8.set_adjustable('box')
    ax8.text(0.01, 0.95, "Ground truth", transform=ax8.transAxes, fontweight="bold",fontsize=figure_config["text"],fontname=figure_config["font"], color='white', ha='left', va='top')
    
    plt.savefig(save_path,bbox_inches='tight',pad_inches=0.01)
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
    plot_organelle(image_index, param=param, save_path=os.path.join(output_dir, 'compare_{}.png'.format(param["organelle"])))
