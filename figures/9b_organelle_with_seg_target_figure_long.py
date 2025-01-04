import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from cell_imaging_utils.image.image_utils import ImageUtils
import os
import cv2
import pandas as pd
from figure_config import figure_config,get_scalebar

params = [
    # {"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","th":0.60,"slice":28},
    {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","th":0.40,"slice":31},
    {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","th":0.2,"slice":37},
    {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","th":0.2,"slice":30},
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

def plot_organelle(image_index, param, save_path):
    """
    Creates a figure with 4 rows and 2 columns:
      - Top 2 rows (spanning both columns): overlay image
      - Bottom 2 rows (2x2 grid): input, GT, noisy pred, original pred
    """
    # Feel free to adjust figsize as needed
    fig = plt.figure(figsize=(5,8.0))
    gs = gridspec.GridSpec(4, 2)
    # Minimal spacing
    gs.update(wspace=0.01, hspace=0.01)

    input_image, mask_image, pred_original, pred_noisy, ground_truth, seg_ground_truth, slice_index = collect_images(param, image_index)
    mask_seg_gt_overlay = overlay_images(input_image, mask_image, seg_ground_truth)

    # Titles
    fig.text(
        0.5, 0.99, param["organelle"],
        fontsize=figure_config["organelle"],
        fontname=figure_config["font"],
        color='black', ha='center', va='top'
    )
    mean_pcc = pd.read_csv(f"{param['model']}/predictions_agg/pcc_resuls.csv")[f"{param['th']:.1f}"].mean()
    mask_size = 1 - pd.read_csv(f"{param['model']}/predictions_agg/mask_size_resuls.csv")[f"{param['th']:.1f}"].mean()
    fig.text(
        0.5, 0.95,
        f"TH={param['th']:.1f}, Mask efficacy={mean_pcc:.2f} [PCC]\nMean noise vol.={mask_size*100:.2f}%",
        fontsize=figure_config["subtitle"],
        fontname=figure_config["font"],
        color='black', ha='center', va='top'
    )
    plt.subplots_adjust(top=0.90)

    # 1) Large overlay subplot spanning rows 0–2, columns 0–2 (the entire top half)
    ax_overlay = plt.subplot(gs[0:2, 0:2])
    ax_overlay.imshow(mask_seg_gt_overlay)
    ax_overlay.axis('off')
    ax_overlay.text(
        0.01, 0.975, "Input Image + Mask overlay + GT segmentation",
        transform=ax_overlay.transAxes,
        fontsize=figure_config["text"],
        fontname=figure_config["font"],
        fontweight="bold",
        color='white',
        ha='left',
        va='top'
    )
    # ax_overlay.add_artist(get_scalebar())

    # 2) Bottom half (2 rows, 2 columns) for the 4 images:
    # Row=2,Col=0 → Input Image
    ax_input = plt.subplot(gs[2, 0])
    ax_input.imshow(input_image, cmap='gray')
    ax_input.axis('off')
    ax_input.text(
        0.01, 0.95, "Input Image",
        transform=ax_input.transAxes,
        fontsize=figure_config["text"],
        fontname=figure_config["font"],
        fontweight="bold",
        color='white',
        ha='left',
        va='top'
    )
    # ax_input.add_artist(get_scalebar())

    # Row=2,Col=1 → Ground Truth
    ax_gt = plt.subplot(gs[2, 1])
    ax_gt.imshow(ground_truth, cmap='gray')
    ax_gt.axis('off')
    ax_gt.text(
        0.01, 0.95, "Ground Truth",
        transform=ax_gt.transAxes,
        fontsize=figure_config["text"],
        fontname=figure_config["font"],
        fontweight="bold",
        color='white',
        ha='left',
        va='top'
    )

    # Row=3,Col=0 → Noisy Prediction
    ax_noisy = plt.subplot(gs[3, 0])
    ax_noisy.imshow(pred_noisy, cmap='gray')
    ax_noisy.axis('off')
    ax_noisy.text(
        0.01, 0.95, "Noisy Prediction",
        transform=ax_noisy.transAxes,
        fontsize=figure_config["text"],
        fontname=figure_config["font"],
        fontweight="bold",
        color='white',
        ha='left',
        va='top'
    )

    # Row=3,Col=1 → Prediction
    ax_pred = plt.subplot(gs[3, 1])
    ax_pred.imshow(pred_original, cmap='gray')
    ax_pred.axis('off')
    ax_pred.text(
        0.01, 0.95, "Prediction",
        transform=ax_pred.transAxes,
        fontsize=figure_config["text"],
        fontname=figure_config["font"],
        fontweight="bold",
        color='white',
        ha='left',
        va='top'
    )

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
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
    plot_organelle(image_index, param=param, save_path=os.path.join(output_dir, 'validation_{}_long.png'.format(param["organelle"])))
