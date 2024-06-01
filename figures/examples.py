import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from cell_imaging_utils.image.image_utils import ImageUtils
import os


params = [
          {"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","slices":[32,32,32,32],"th":0.6},
        #   {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5"},
        #   {"organelle":"Endoplasmic-reticulum","model":"../mg_model_er_13_05_24_1.5"},
        #   {"organelle":"Golgi","model":"../mg_model_golgi_13_05_24_1.5"},
        #   {"organelle":"Actomyosin-bundles","model":"../mg_model_bundles_13_05_24_1.5"}
          
        #   {"organelle":"Mitochondria","model":"../unet_model_22_05_22_mito_128"},
        #   {"organelle":"Nuclear-envelope","model":"../unet_model_22_05_22_ne_128"},
        #   {"organelle":"Microtubules","model":"../unet_model_22_05_22_microtubules_128"},
        #   {"organelle":"Actin-filaments","model":"../unet_model_22_05_22_actin_128"},
          ]
# Function to overlay mask on input image
def overlay_mask(input_image, mask_image, alpha=0.5):
    mask_image = mask_image / np.max(mask_image)
    # Combine input image and mask
    overlay = input_image * (1 - alpha) + mask_image * alpha
    return overlay

# Create sample images
def collect_images(param,image_index):
    input_image = ImageUtils.image_to_ndarray(ImageUtils.imread("{}/predictions_agg/input_{}.tiff".format(param["model"],image_index)))[:,:,param["slices"][image_index]]
    mask_image = ImageUtils.image_to_ndarray(ImageUtils.imread("{}/predictions_agg/{}/mask_{}.tiff".format(param["model"],param["th"],image_index)))[:,:,param["slices"][image_index]]
    prediction_original = ImageUtils.image_to_ndarray(ImageUtils.imread("{}/predictions_agg/unet_prediction_{}.tiff".format(param["model"],image_index)))[:,:,param["slices"][image_index]]
    prediction_noisy = ImageUtils.image_to_ndarray(ImageUtils.imread("{}/predictions_agg/{}/noisy_unet_prediction_{}.tiff".format(param["model"],param["th"],image_index)))[:,:,param["slices"][image_index]]
    ground_truth = ImageUtils.image_to_ndarray(ImageUtils.imread("{}/predictions_agg/target_{}.tiff".format(param["model"],image_index)))[:,:,param["slices"][image_index]]
    return input_image, mask_image, prediction_original, prediction_noisy, ground_truth

# Resize images to fit the layout
def resize_image(image, size=(int(312*1.5), int(462*1.5))):
    return resize(image, size, anti_aliasing=True)

# Create plot for a single organelle
def plot_organelle(examples_per_organelle, param, save_path):
    fig, axes = plt.subplots(4, examples_per_organelle, figsize=(9, 6), gridspec_kw={'hspace': 0.005, 'wspace': 0.001})
    
    for j in range(examples_per_organelle):
        input_image, mask_image, prediction_original, prediction_noisy, ground_truth = collect_images(param,j)
        
        overlay_image = overlay_mask(input_image, mask_image)
        
        overlay_image = resize_image(overlay_image)
        prediction_noisy = resize_image(prediction_noisy)
        prediction_original = resize_image(prediction_original)
        ground_truth = resize_image(ground_truth)
        
        axes[0, j].imshow(overlay_image)
        if j == 0:
            axes[0, j].set_ylabel('Input+Mask', fontsize=10, rotation=45, labelpad=30)
        
        axes[1, j].imshow(prediction_noisy, cmap='gray')
        if j == 0:
            axes[1, j].set_ylabel('Prediction Noisy', fontsize=10, rotation=45, labelpad=30)
        
        axes[2, j].imshow(prediction_original, cmap='gray')
        if j == 0:
            axes[2, j].set_ylabel('Prediction Original', fontsize=10, rotation=45, labelpad=30)
        
        axes[3, j].imshow(ground_truth, cmap='gray')
        if j == 0:
            axes[3, j].set_ylabel('Ground Truth', fontsize=10, rotation=45, labelpad=30)

    # Hide the y-axis lines and ticks but keep the labels
    for ax in axes.flat:
        ax.get_xaxis().set_visible(False)
        ax.yaxis.label.set_visible(True)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)   
           
    fig.suptitle("{} Mean PCC={}, TH={}".format(param["organelle"],param["th"]), fontsize=16,y=0.92) 
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# Create the first figure with a 3x2 grid
def create_figure1(num_organelles, organelle_names, output_dir):
    fig, big_axes = plt.subplots(3, 2, figsize=(12, 12), gridspec_kw={'hspace': 0, 'wspace': 0})

    # Load each organelle's plot and add to the grid
    for i in range(3):
        for j in range(2):
            index = i * 2 + j
            if index < num_organelles:
                ax = big_axes[i, j]
                ax.axis('off')
                image_path = os.path.join(output_dir, f"{organelle_names[index]}.png")
                if os.path.exists(image_path):
                    img = plt.imread(image_path)
                    ax.imshow(img)
            else:
                big_axes[i, j].axis('off')

    fig.suptitle('Mask Interpreter Multiple Examples per Organelle - Part 1', fontsize=16, y=0.9)  # Adjust the y parameter to control the padding
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(output_dir, "multiple_organelles_part1.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

# Create the second figure with a 2x2 grid plus 1 in the third row
def create_figure2(num_organelles, organelle_names, output_dir):
    fig, big_axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'hspace': 0, 'wspace': 0})

    # Load each organelle's plot and add to the grid
    for i in range(1):
        for j in range(2):
            index = 6 + i * 2 + j
            if index < num_organelles:
                ax = big_axes[i, j]
                ax.axis('off')
                image_path = os.path.join(output_dir, f"{organelle_names[index]}.png")
                if os.path.exists(image_path):
                    img = plt.imread(image_path)
                    ax.imshow(img)
            else:
                big_axes[i, j].axis('off')

    # Place the last organelle in the center of the third row
    if num_organelles > 8:
        ax = big_axes[1, 0]
        ax.axis('off')
        image_path = os.path.join(output_dir, f"{organelle_names[8]}.png")
        if os.path.exists(image_path):
            img = plt.imread(image_path)
            ax.imshow(img)
        big_axes[1, 1].axis('off')  # Hide the second cell in the third row

    fig.suptitle('Mask Interpreter Multiple Examples per Organelle - Part 2', fontsize=16, y=0.9)  # Adjust the y parameter to control the padding
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(output_dir, "multiple_organelles_part2.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

# Directory to save individual plots
output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)

# Define organelle names
organelle_names = [f'Organelle {i+1}' for i in range(9)]

# Generate and save individual organelle plots
for param in params:
    plot_organelle(examples_per_organelle=4, param = param, save_path=os.path.join(output_dir, '{}.png'.format(param["organelle"])))

# Create the first figure with the first 6 organelles
create_figure1(num_organelles=6,organelle_names, output_dir=output_dir)

# Create the second figure with the remaining organelles
create_figure2(num_organelles=3,organelle_names, output_dir=output_dir)
