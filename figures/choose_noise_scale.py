#This script is used to choose noise std for each organelle, 
# we noise the entire image with different noise stds and choose the one 
# that reduce PCC below 0.1

import matplotlib.pyplot as plt
import numpy as np
from dataset import DataGen
import global_vars as gv
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from mg_analyzer import find_noise_scale

params = [{"organelle":"Nuclear-envelope","model":"./unet_model_22_05_22_ne_128"},
        #   {"organelle":"Golgi","model":"./unet_model_22_05_22_golgi_128"},
        #   {"organelle":"Microtubules","model":"./unet_model_22_05_22_microtubules_128"},
        #   {"organelle":"Endoplasmic-reticulum","model":"./unet_model_22_05_22_er_128"},
        #   {"organelle":"Plasma-membrane","model":"./unet_model_22_05_22_membrane_128"},
        #   {"organelle":"Actin-filaments","model":"./unet_model_22_05_22_actin_128"},
        #   {"organelle":"Nucleolus-(Granular-Component)","model":"./unet_model_22_05_22_ngc_128"},
        #   {"organelle":"Mitochondria","model":"./unet_model_22_05_22_mito_128"},
        #   {"organelle":"Actomyosin-bundles","model":"./unet_model_22_05_22_bundles_128"}
          ]

gv.input = "channel_signal"
gv.target = "channel_target"
weighted_pcc = False

def plot_noise_scale_analysis():
    # Enhanced setup for 9 subplots, each with additional annotations for the first x-value where PCC < 0.1
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))  # 3x3 grid of subplots

    for i, ax in enumerate(axs.flatten()):
        # load PCC data for each subplot
        pcc_data = DatasetMetadataSCV("{}/noise_pcc_resuls.csv".format(params[i]["model"]))
        noise_std = pcc_data.data.columns
        pcc = pcc_data.data.mean(axis=1)
        
        # Plotting
        ax.plot(noise_std, pcc, label='Model Prediction Quality')
        ax.axhline(y=0.1, color='r', linestyle='--', label='Threshold: PCC=0.1')
        ax.set_title(params[i]["organelle"])
        ax.set_xlabel('Noise Standard Deviation')
        ax.set_ylabel('PCC')
        
        # Find the first x-value where PCC < 0.1 and annotate
        below_threshold = noise_std[pcc < 0.1]
        if below_threshold.size > 0:  # Check if there's any value below the threshold
            first_below = below_threshold[0]
            ax.axvline(x=first_below, color='g', linestyle=':', label=f'Critical Noise STD: {first_below:.2f}')
            ax.annotate(f'{first_below:.2f}', xy=(first_below, 0.1), xytext=(first_below, 0.15),
                        textcoords='data', arrowprops=dict(facecolor='black', arrowstyle='->'),
                        horizontalalignment='right', verticalalignment='top')
        
        ax.legend()
        ax.grid(True)

    # Main title and layout adjustments
    plt.suptitle('Effect of Noisy Input on Model Prediction by Standard Deviation', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
    plt.savefig("./figures/find_noise_scale.png")


for param in params:
    ds_path = "/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/{}/image_list_test.csv".format(param["organelle"])
    dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
    find_noise_scale(dataset,model_path=param["model"],weighted_pcc=weighted_pcc)
plot_noise_scale_analysis()


