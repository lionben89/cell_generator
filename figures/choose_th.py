#This script is used to choose noise std for each organelle, 
# we noise the entire image with different noise stds and choose the one 
# that reduce PCC below 0.2

import matplotlib.pyplot as plt
import numpy as np
from dataset import DataGen
import global_vars as gv
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from mg_analyzer import analyze_th

params = [
          {"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5"},
        #   {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5"},
        #   {"organelle":"Endoplasmic-reticulum","model":"../mg_model_er_13_05_24_1.5"},
        #   {"organelle":"Golgi","model":"../mg_model_golgi_13_05_24_1.5"},
        #   {"organelle":"Actomyosin-bundles","model":"../mg_model_bundles_13_05_24_1.5"}
          
        #   {"organelle":"Mitochondria","model":"../unet_model_22_05_22_mito_128"},
        #   {"organelle":"Nuclear-envelope","model":"../unet_model_22_05_22_ne_128"},
        #   {"organelle":"Microtubules","model":"../unet_model_22_05_22_microtubules_128"},
        #   {"organelle":"Actin-filaments","model":"../unet_model_22_05_22_actin_128"},
          ]

gv.input = "channel_signal"
gv.target = "channel_target"
weighted_pcc = False


def plot_th_analysis():
    # Enhanced setup for 9 subplots, each with additional annotations for the first x-value where PCC < 0.9
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))  # 3x3 grid of subplots

    for i, ax in enumerate(axs.flatten()):
        try:
            # load PCC data for each subplot
            csv_path = "{}/predictions_agg/pcc_resuls.csv".format(params[i]["model"])
            pcc_data = DatasetMetadataSCV(csv_path,csv_path)
            ths = np.array(list(map(lambda val: float('{:.2f}'.format(float(val))), pcc_data.data.columns[1:])))
            pcc = pcc_data.data.mean(axis=0)[1:]
            
            # Plotting
            ax.plot(ths, pcc, label='Model Prediction Quality')
            ax.axhline(y=0.9, color='r', linestyle='--', label='Threshold: PCC=0.9')
            ax.set_title(params[i]["organelle"])
            ax.set_xlabel('TH')
            ax.set_ylabel('PCC')
            # Find the first x-value where PCC < 0.2 and annotate
            over_threshold = ths[pcc > 0.9]
            if over_threshold.size > 0:  # Check if there's any value below the threshold
                last_over = over_threshold[-1]
                ax.axvline(x=last_over, color='g', linestyle=':', label=f'Critical TH: {last_over}')
            
            ax.legend()
        except Exception as e:
            print("data for subplot {} not exist".format(i))

    # Main title and layout adjustments
    plt.suptitle('Effect of TH the mask on Model Prediction from Noisy Input', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
    plt.savefig("../figures/find_th.png")


for param in params:
    print(param["organelle"])
    ds_path = "/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/{}/image_list_test.csv".format(param["organelle"])
    dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
    analyze_th(dataset,"agg",mask_image=None,manual_th="full",save_image=4,save_histo=False,weighted_pcc = weighted_pcc, model_path=param["model"],model=None,compound=None,images=range(10))
plot_th_analysis()


