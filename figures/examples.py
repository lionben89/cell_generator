import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from scipy.stats import entropy
from cell_imaging_utils.image.image_utils import ImageUtils
import os


params = [
          {"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","th":0.60,"slices":[30,36,27,36]},
          {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","th":0.40,"slices":[32,24,34,29]},
          {"organelle":"Endoplasmic-reticulum","model":"../mg_model_er_13_05_24_1.5","th":0.40,"slices":[22,42,42,24]},
          {"organelle":"Golgi","model":"../mg_model_golgi_13_05_24_1.5","th":0.30,"slices":[44,35,45,46]},
          {"organelle":"Actomyosin-bundles","model":"../mg_model_bundles_13_05_24_1.0","th":0.02,"slices":[54,32,32,32]},
          {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","th":0.2,"slices":[44,28,36,25]},
          {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","th":0.2,"slices":[42,32,33,18]},
          {"organelle":"Microtubules","model":"../mg_model_microtubules_13_05_24_1.5","th":0.10,"slices":[27,24,28,20]},
          {"organelle":"Actin-filaments","model":"../mg_model_actin_13_05_24_1.5","th":0.20,"slices":[56,48,36,55]},
          ]

# Function to overlay mask on input image with a turquoise color
def overlay_mask(input_image, mask_image, alpha=0.25):
    # Create an RGB version of the gray input image
    colored_input = np.stack([input_image] * 3, axis=-1)
    # Define the turquoise color (normalized RGB values)
    turquoise_color = np.array([64, 224, 208]) / 255.0
    mask_rgb = np.stack([mask_image * turquoise_color[0],
                         mask_image * turquoise_color[1],
                         mask_image * turquoise_color[2]], axis=-1)
    # Combine input image and mask
    overlay = colored_input * (1 - alpha) + mask_rgb * alpha
    return (overlay*255.0).astype(int)

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
    most_informative_slices = {key: stack[best_slice_index, :, :, 0] for key, stack in stacks.items()}

    return most_informative_slices["input"], most_informative_slices["mask"], most_informative_slices["prediction"], most_informative_slices["noisy_prediction"], most_informative_slices["ground_truth"],best_slice_index

# Resize images to fit the layout
def resize_image(image, size=(int(312*1.5), int(462*1.5))):
    return resize(image, size, anti_aliasing=True)

# Create plot for a single organelle
def plot_organelle(examples_per_organelle, param, save_path):
    fig, axes = plt.subplots(4, examples_per_organelle, figsize=(9.75, 8), gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    
    for j in range(examples_per_organelle):
        input_image, mask_image, prediction_original, prediction_noisy, ground_truth, slice_index = collect_images(param, j)
        
        # Display the overlay image where input and mask are combined
        overlay_image = overlay_mask(input_image, mask_image)
        axes[0, j].imshow(overlay_image)
        title = axes[0, j].set_title(f'Slice {slice_index}', fontsize=10, pad=10)  # Increase pad to adjust space below title
        title.set_position([0.5, 1.05])  # Adjust the title position

        # Display prediction images and ground truth
        axes[1, j].imshow(prediction_noisy, cmap='gray')
        axes[2, j].imshow(prediction_original, cmap='gray')
        axes[3, j].imshow(ground_truth, cmap='gray')
        
        # Set y-axis labels for the first column
        if j == 0:
            axes[0, j].set_ylabel('Input+Mask', fontsize=10, rotation=45, labelpad=30)
            axes[1, j].set_ylabel('Prediction Noisy', fontsize=10, rotation=45, labelpad=30)
            axes[2, j].set_ylabel('Prediction Original', fontsize=10, rotation=45, labelpad=30)
            axes[3, j].set_ylabel('Ground Truth', fontsize=10, rotation=45, labelpad=30)

    # Adjust axis visibility
    for ax in axes.flat:
        ax.get_xaxis().set_visible(False)
        ax.yaxis.label.set_visible(True)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)   
    
    fig.suptitle(f"{param['organelle']}, TH={param['th']}", fontsize=16, y=0.92)
    plt.subplots_adjust(top=0.85)  # Adjust top spacing to accommodate subplot titles
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# Create the first figure with a 3x2 grid
def create_figure1(num_organelles, organelle_names, output_dir):
    fig, big_axes = plt.subplots(3, 2, figsize=(10, 12), gridspec_kw={'hspace': 0, 'wspace': 0})

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
    fig, big_axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0, 'wspace': 0})

    # Load each organelle's plot and add to the grid
    for i in range(1):
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

    # Place the last organelle in the center of the third row
    if num_organelles % 2 ==1:
        ax = big_axes[1, 0]
        ax.axis('off')
        image_path = os.path.join(output_dir, f"{organelle_names[-1]}.png")
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
organelle_names = []
# Generate and save individual organelle plots
for param in params:
    print(param["organelle"])
    organelle_names.append(param["organelle"])
    plot_organelle(examples_per_organelle=4, param = param, save_path=os.path.join(output_dir, '{}.png'.format(param["organelle"])))

# Create the first figure with the first 6 organelles
create_figure1(num_organelles=6,organelle_names=organelle_names[:6], output_dir=output_dir)

# Create the second figure with the remaining organelles
create_figure2(num_organelles=3,organelle_names=organelle_names[6:], output_dir=output_dir)
