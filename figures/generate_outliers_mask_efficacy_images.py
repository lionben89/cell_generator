import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
from dataset import DataGen
import global_vars as gv
from mg_analyzer import analyze_th
from utils import *

# Define the path to your CSV files
path = "/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/**/image_list_with_metadata__with_efficacy_scores_full.csv"

# Print the path pattern
print(f"Using path pattern: {path}")

# Use glob to get all the CSV file paths
csv_files = glob.glob(path, recursive=True)

# Print the found files
print(f"Found {len(csv_files)} CSV files:")
for file in csv_files:
    print(file)

# Read each CSV file into a DataFrame and store them in a list
dfs = [pd.read_csv(file) for file in csv_files]

# Check if any DataFrames were created
if not dfs:
    print("No DataFrames were created. Please check the CSV files.")
else:
    # Concatenate all DataFrames
    metadata_with_efficacy_scores_df = pd.concat(dfs, ignore_index=True)
    print("metadata_with_efficacy_scores_df # FOVS:{}".format(metadata_with_efficacy_scores_df.shape[0]))
    
    metadata_with_efficacy_scores_df.to_csv("/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/image_list_with_metadata__with_efficacy_scores_full_all.csv")

# List of columns with the results to plot
params = [
        #   {"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","noise":1.5,"th":0.60},
        #   {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","noise":1.5,"th":0.40},
        #   {"organelle":"Endoplasmic-reticulum","model":"../mg_model_er_13_05_24_1.5","noise":1.5,"th":0.40},
        #   {"organelle":"Golgi","model":"../mg_model_golgi_13_05_24_1.5","noise":1.5,"th":0.30},
        #   {"organelle":"Actomyosin-bundles","model":"../mg_model_bundles_13_05_24_1.0","noise":1.0,"th":0.02},
        #   {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","noise":1.5,"th":0.20},
          {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","noise":1.0,"th":0.20},
          {"organelle":"Microtubules","model":"../mg_model_microtubules_13_05_24_1.5","noise":1.5,"th":0.10},
          {"organelle":"Actin-filaments","model":"../mg_model_actin_13_05_24_1.5","noise":1.5,"th":0.20},
          ]
weighted_pcc = False
# Function to plot box plots for each column and result
def generate_low_efficacy_images(metadata_with_efficacy_scores_df, params):
    for param in tqdm(params):
        ## Collect training and testing data for the model
        control_df = pd.read_csv("/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/{}/image_list_with_metadata__with_efficacy_scores_full.csv".format(param["organelle"]))
        control_values = control_df[param["model"]].dropna()
  
        # Calculate the lower and upper bounds for the control group without outliers
        q1 = np.percentile(control_values, 25)
        q3 = np.percentile(control_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        # upper_bound = q3 + 1.5 * iqr
        results_save_path = "/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/outliers/{}".format(param["model"].split('/')[-1])
        create_dir_if_not_exist(results_save_path)
        outliers_ds_path = "{}/outliers.csv".format(results_save_path)
        outliers_df = metadata_with_efficacy_scores_df[metadata_with_efficacy_scores_df[param["model"]] < lower_bound].sort_values(by=param["model"])
        outliers_df.to_csv(outliers_ds_path,index=False)
        dataset = DataGen(outliers_ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False,image_path_col='combined_image_storage_path')
        print("# images in dataset:",dataset.df.data.shape[0])
        analyze_th(dataset,"regular",mask_image=None,manual_th=param["th"],save_image=50,save_histo=False,weighted_pcc = weighted_pcc, model_path=param["model"],model=None,compound=None,images=range(min(50,dataset.df.data.shape[0])),noise_scale=param["noise"],results_save_path=results_save_path)
        
generate_low_efficacy_images(metadata_with_efficacy_scores_df, params)