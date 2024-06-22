#This script is used to choose noise std for each organelle, 
# we noise the entire image with different noise stds and choose the one 
# that reduce PCC below 0.2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import DataGen
import global_vars as gv
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from mg_analyzer import analyze_th

params = [
          {"organelle":"Nucleolus-(Granular-Component)","model":"../mg_model_ngc_13_05_24_1.5","noise":1.5},
          {"organelle":"Plasma-membrane","model":"../mg_model_membrane_13_05_24_1.5","noise":1.5},
          {"organelle":"Endoplasmic-reticulum","model":"../mg_model_er_13_05_24_1.5","noise":1.5},
          {"organelle":"Golgi","model":"../mg_model_golgi_13_05_24_1.5","noise":1.5},
          {"organelle":"Actomyosin-bundles","model":"../mg_model_bundles_13_05_24_1.0","noise":1.0},
          {"organelle":"Mitochondria","model":"../mg_model_mito_13_05_24_1.5","noise":1.5},
          {"organelle":"Nuclear-envelope","model":"../mg_model_ne_13_05_24_1.0","noise":1.0},
          {"organelle":"Microtubules","model":"../mg_model_microtubules_13_05_24_1.5","noise":1.5},
          {"organelle":"Actin-filaments","model":"../mg_model_actin_13_05_24_1.5","noise":1.5},
          ]
base_dir = "/sise/assafzar-group/assafzar/full_cells_fovs/"
gv.input = "channel_signal"
gv.target = "channel_target"
weighted_pcc = False

# Create a list of organelles from the parameters
organelles = [
              "Actin-filaments",
              "Actomyosin-bundles",
              "Adherens-junctions",
              "Desmosomes",
              "Endoplasmic-reticulum",
              "Endosomes",
              "Gap-junctions",
              "Golgi",
              "Lysosome",
            #   "Matrix-adhesions",
              "Microtubules",
              "Mitochondria",
              "Nuclear-envelope",
              "Nucleolus-(Dense-Fibrillar-Component)",
              "Nucleolus-(Granular-Component)",
              "Peroxisomes",
              "Plasma-membrane",
              "Tight-junctions"
              ]
for organelle in organelles:
    images_list1 = "{}/{}/{}.csv".format(base_dir,organelle,organelle)
    images_list2 = "{}/train_test_list/{}/image_list_train.csv".format(base_dir)
    images_list3 = "{}/train_test_list/{}/image_list_test.csv".format(base_dir)
    images_df = pd.read_csv(images_list1)
    images_df2 = pd.read_csv(images_list2)
    images_df3 = pd.read_csv(images_list3)
    # Merge df with df1 and df2 based on the appropriate columns
    df_merged = images_df.merge(images_df2, left_on='combined_image_storage_path', right_on='path_tiff', how='left') \
                .merge(images_df3, left_on='combined_image_storage_path', right_on='path_tiff', how='left', suffixes=('_df2', '_df3'))

    # Combine columns from df1 and df2
    for col in images_df2.columns:
        if col != 'path_tiff':
            df_merged[col] = df_merged[f'{col}_df1'].combine_first(df_merged[f'{col}_df2'])
            df_merged.drop(columns=[f'{col}_df1', f'{col}_df2'], inplace=True)

    # Fill NaN values if needed
    df_merged.fillna('', inplace=True)

    # Drop redundant 'path_tiff' columns
    df_merged.drop(columns=['path_tiff'], inplace=True)

    # Print the resulting dataframe
    print(df_merged.columns)
    print(df_merged.head(10))
    
    # for param in params:
    #     print("model:",param["organelle"])
    #     ds_path = "/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/{}/image_list_test.csv".format(param["organelle"])
    #     dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
    #     print("# images in dataset:",dataset.df.data.shape[0])
    #     analyze_th(dataset,"agg",mask_image=None,manual_th="full",save_image=5,save_histo=False,weighted_pcc = weighted_pcc, model_path=param["model"],model=None,compound=None,images=range(min(10,dataset.df.data.shape[0])),noise_scale=param["noise"])



