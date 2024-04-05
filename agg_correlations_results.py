import pandas as pd
import os
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

source_to_clone = {\
    "Actin-filaments":'79',"Actomyosin-bundles":'80',"Adherens-junctions":'67',"Desmosomes":'65',"Endoplasmic-reticulum":'55',\
        "Endosomes":'35', "Gap-junctions":'16',"Golgi":'44', "Lysosome":'37', "Matrix-adhesions":'50', "Microtubules":'105',\
            "Mitochondria":'27', "Nuclear-envelope":'210', "Nucleolus-(Dense-Fibrillar-Component)":'6',\
                "Nucleolus-(Granular-Component)":'50',"Peroxisomes":'115',"Plasma-membrane":'91',"Tight-junctions":'20'\
                        }

def calculate_column_averages(df):
    if 'importance_in_organelle' in df.columns:
        df = df.drop('importance_in_organelle',axis=1)
    df_mean = df.iloc[:, 1:].mean()
    df_std = df.iloc[:, 1:].std()

    column_averages = {}

    for column, average in df_mean.items():
        if np.isnan(average):
            column_averages[column] = -1.0
        else:
            column_averages[column] = average
        
    for column, std in df_std.items():  
        if np.isnan(std):      
            column_averages["{}_std".format(column)] = -1.0
        else:
            column_averages["{}_std".format(column)] = std

    return column_averages

def calculate_p_values(df, column_pairs):
    df = df.iloc[:, 1:]
    p_values = {}

    for column_pair in column_pairs:
        col1, col2 = column_pair
        p_value = stats.ttest_ind(df[col1], df[col2],alternative='less').pvalue
        p_values[f'P-Value ({col1}, {col2})'] = p_value

    return p_values

def write_averages_to_csv(df_output, output_file, drug, v):  
    global mean_df 
    global mode 
    df_output.to_csv("{}.csv".format(output_file), index=False)
    
    # Create a DataFrame with all combinations of sources and targets
    all_combinations = pd.DataFrame([(s, t) for s in pertrubed_organelles for t in df_output['target'].unique()], columns=['source', 'target'])

    # Merge the original DataFrame with the all_combinations DataFrame to get missing values as NaN
    merged_df = pd.merge(all_combinations, df_output, on=['source', 'target'], how='left')

    # Pivot the merged DataFrame to create a matrix of source-target pairs and their values
    if mode == "correlations":
        pivot_df = merged_df.pivot(index='source', columns='target', values='pcc') #'pcc_wo_organelle_pixels_wo_mask'
        # pivot_df = 1/pivot_df
        # pivot_df = pivot_df < 0.05
    else:
        merged_df = merged_df.drop_duplicates(['source','target'])
        pivot_df = merged_df.pivot(index='source', columns='target', values='pcc')
    # Create the heatmap using Seaborn
    plt.figure(figsize=(16, 8))
    if mode == "correlations":
        # sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, vmin=0.0, vmax=2.0)
        sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, vmin=0.3, vmax=0.95)
        plt.title('Heatmap of Source-Target Pairs')
    else:
        sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, vmin=0.0, vmax=1.0)
        plt.title('Prediction performence')
    plt.xlabel('Target')
    plt.ylabel('Source')
    
    # Rotate the x-axis labels horizontally
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.xticks(rotation=0)

    # Save the plot as an image file (e.g., PNG)
    if mode == "correlations":
        plt.savefig("{}_importance.png".format(output_file))
    else:
        plt.savefig("{}_pred_vs_gt.png".format(output_file))
    
    # Filter the DataFrame to include only the rows with sources present in the list 'l_sources'
    df_filtered = df_output[(df_output['source'].isin(pertrubed_organelles)) & (df_output['pcc'] > 0.06)]

    # Group by target and calculate the mean of the 'value' column for each target
    if mode == "correlations":
        value = 'pcc'
    else:    
        value = 'pcc'
    target_means = df_filtered.groupby('target')[value].mean().reset_index()
    target_std = df_filtered.groupby('target')["{}_std".format(value)].mean().reset_index()

    # Rename the 'value' column to 'mean_value' for clarity
    target_means.rename(columns={value: 'mean_pcc'}, inplace=True)
    target_std.rename(columns={value: 'pcc_std'}, inplace=True)

    if drug is None:
        if v is None:
            drug_t = "DMSO"
        else:
            drug_t = v
    else:
        drug_t = drug
    
    target_means['drug'] = drug_t
    target_std['drug'] = drug_t
    
    # Append the target means to the mean DataFrame
    target_data = pd.merge(target_means, target_std, on=['target', 'drug'])
    mean_df = mean_df.append(target_data, ignore_index=True)

def write_all_to_csv(df_output, output_file, drug, v):   
    global mode 
    df_output.to_csv("{}.csv".format(output_file), index=False)
    
    # Create a DataFrame with all combinations of sources and targets
    all_combinations = pd.DataFrame([(s, t) for s in pertrubed_organelles for t in df_output['target'].unique()], columns=['source', 'target'])

    # Merge the original DataFrame with the all_combinations DataFrame to get missing values as NaN
    merged_df = pd.merge(all_combinations, df_output, on=['source', 'target'], how='left')
    
    # Add 'clone' column by transforming 'source' using the dictionary
    merged_df['clone'] = merged_df['source'].map(source_to_clone)
    
    # Get unique values from 'target' column
    targets = merged_df['target'].unique()
    n_targets = len(targets)
    sources = merged_df['source'].unique()

    # Create a grid of subplots
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4*n_targets))

    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        # Filter data for the current target
        subset_df = merged_df[merged_df['target'] == target]
        
        # Boxplot
        # sns.boxplot(data=subset_df, x='pcc', y='source', ax=ax, palette='Set2')
        
        # Swarmplot or Stripplot
        if mode == "correlations":
            sns.swarmplot(data=subset_df, x='pcc', y='clone', ax=ax, color=".1")
            
            # Calculate IQR and add dashed lines for the entire target data
            Q1 = subset_df['pcc'].quantile(0.25)
            Q3 = subset_df['pcc'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.0 * IQR
            upper_bound = Q3 + 1.0 * IQR
            
            ax.axvline(lower_bound, color='r', linestyle='--', label='IQR-')
            ax.axvline(upper_bound, color='r', linestyle='--', label='IQR+')
        else:
            sns.boxplot(data=subset_df, x='clone', y='pcc', color="skyblue",ax=ax)
        
        ax.set_title(target)
        if mode == "correlations":
            ax.set_ylabel("Clone")        
            ax.set_xlabel("Mask explanation efficacy [PCC]")
        else:
            ax.set_xlabel("Clone")
            ax.set_ylabel("Prediction score VS ground truth [PCC]")        
            
        

    # Only one legend
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=len(sources))

    plt.tight_layout()
    if mode == "correlations":
        plt.savefig("{}_dist_corrlations.png".format(output_file)) 
    else:
        plt.savefig("{}_dist_predictions.png".format(output_file)) 


method = "mean"
# mode="correlations" ##
mode = "predictions"

# Define pairs of columns for p-value calculation
column_pairs = [('pcc_wo_organelle_pixels', 'pcc_wo_organelle_pixels_wo_mask'),('pcc_wo_organelle_pixels','pcc_random_flip'),('pcc_wo_organelle_pixels_wo_mask','pcc_random_flip'),('pcc_wo_organelle_pixels','pcc_random_pixels'),('pcc_wo_organelle_pixels_wo_mask','pcc_random_pixels')]  # Update with your desired column pairs

if mode == "correlations":
    models = [{"model":"mg_model_dna_10_06_22_5_0_dnab3",'organelle':"DNA"},{"model":"mg_model_er_10_06_22_5_0_mlw_0.17",'organelle':"Endoplasmic-reticulum"},{'model':'mg_model_microtubules_10_06_22_5_0_new','organelle':"Microtubules"},{'model':'mg_model_golgi_10_06_22_5_0_new','organelle':"Golgi"},{'model':'mg_model_actin_10_06_22_5_0_new','organelle':"Actin-filaments"},{'model':'mg_model_membrane_10_06_22_5_0_new','organelle':"Plasma-membrane"},{'model':'mg_model_ne_10_06_22_5_0_new','organelle':"Nuclear-envelope"},{'model':'mg_model_ngc_10_06_22_5_0_new','organelle':"Nucleolus-(Granular-Component)"},{'model':'mg_model_mito_10_06_22_5_0_new','organelle':"Mitochondria"}]
else:
    models = [\
        # {"model":"./unet_model_22_05_22_mito_128","organelle":"Mitochondria"},\
            # {"model":"./unet_model_22_05_22_membrane_128","organelle":"Plasma-membrane"},\
            # {"model":"./unet_model_22_05_22_ne_128","organelle":"Nuclear-envelope"},\
            # {"model":"./unet_model_22_05_22_ngc_128","organelle":"Nucleolus-(Granular-Component)"},\
    {'model':'unet_model_22_05_22_dna_128b','organelle':"DNA"},\
        {'model':'unet_model_22_05_22_er_128','organelle':"Endoplasmic-reticulum"},\
            {'model':'unet_model_22_05_22_microtubules_128','organelle':"Microtubules"},\
                {'model':'unet_model_22_05_22_golgi_128','organelle':"Golgi"},\
                    {'model':'unet_model_22_05_22_actin_128','organelle':"Actin-filaments"},\
                        {'model':'unet_model_22_05_22_bundles_128','organelle':"Actomyosin-bundles"},\
                            {'model':'unet_model_22_05_22_tj_128','organelle':"Tight-junctions"}]

organelles=[{'name':"Nuclear-envelope",'location':"Nuclear periphry"},{'name':"Endoplasmic-reticulum",'location':"Cytoplasm"},{'name':"Golgi",'location':"Cytoplasm"},{'name':"Plasma-membrane",'location':"Cell periphery"},{'name':"Nucleolus-(Dense-Fibrillar-Component)",'location':"Nuclear"},{'name':"Nucleolus-(Granular-Component)",'location':"Nuclear"},{'name':"Microtubules",'location':"Cytoplasm"}\
    ,{'name':"Tight-junctions",'location':"Apical domain"},{'name':"Mitochondria",'location':"Cytoplasm"},{'name':"Actomyosin-bundles",'location':"Cell periphery"},{'name':"Actin-filaments",'location':"Cell periphery"},{'name':"Desmosomes",'location':"Apical domain"},{'name':"Lysosome",'location':"Cytoplasm"},{'name':"Adherens-junctions",'location':"Apical domain"},{'name':"Gap-junctions",'location':"Apical domain"},{'name':"Matrix-adhesions",'location':"Basal domain"},{'name':"Peroxisomes",'location':"Cytoplasm"},{'name':"Endosomes",'location':"Cytoplasm"}]

# pertrubed_organelles = ["Microtubules","Actin-filaments","Golgi","Lysosome","Endoplasmic-reticulum","Tight-junctions","Actomyosin-bundles"]
pertrubed_organelles = []
for organelle in organelles:
    pertrubed_organelles.append(organelle['name'])

drugs = [None,"s-Nitro-Blebbistatin","Rapamycin","Paclitaxol","Staurosporine","Brefeldin"]
first_v = True
v_df = {}
main_df = pd.DataFrame()
v_main_df = pd.DataFrame()
mean_df = pd.DataFrame(columns=['target', 'mean_pcc', 'pcc_std' 'drug'])
for v in [None,"Vehicle"]:
    for drug in drugs:
        # Process each CSV file and calculate column averages and p-values
        if v == "Vehicle" and first_v:
            output_data = []
            headers = []
            first_v = False
        else:
            if v != "Vehicle":
                output_data = []
                headers = []
        for model in models:
            for organelle_item in organelles:
                try:
                    organelle = organelle_item['name']
                    if (model["organelle"] == "DNA") or model["organelle"] == organelle or mode=="correlations":
                        if drug is not None:
                            organelle1 = "{}_{}".format(organelle.replace("-", " "),drug)
                            if v is None:
                                organelle2 = organelle1
                            else:
                                organelle2 = "{}_{}".format(organelle.replace("-", " "),v)
                        else:
                            if v is None:
                                organelle1 = organelle
                                if mode == "correlations":
                                    organelle2 = "{}_{}".format(organelle.replace("-", " "),v)
                                else:   
                                    organelle1 = organelle.replace("-", " ")  
                                    organelle2 = "{}_{}".format(organelle,v)
                                if mode == "correlations":
                                    csv_file = "/sise/home/lionb/{}/predictions_correlations_constant/{}/corr_results_{}_organelle_precent_pixels_0.0_in_image.csv".format(model['model'],organelle1,organelle2)
                                else:
                                    csv_file = "/sise/home/lionb/{}/predictions/{}/predictions_results_{}_in_image.csv".format(model['model'],organelle1,organelle2)
                                if not os.path.exists(csv_file):  
                                    organelle2 = "{}".format(organelle)   
                                    if mode == "correlations":                        
                                        csv_file = "/sise/home/lionb/{}/predictions_correlations_constant/{}/corr_results_{}_organelle_precent_pixels_0.0_in_image.csv".format(model['model'],organelle1,organelle2)
                                    else:
                                        csv_file = "/sise/home/lionb/{}/predictions/{}/predictions_results_{}_in_image.csv".format(model['model'],organelle1,organelle2)
                                    if not os.path.exists(csv_file): 
                                        organelle2 = "{}_{}".format(organelle.replace("-", " "),None)       
                                        if mode == "correlations":                    
                                            csv_file = "/sise/home/lionb/{}/predictions_correlations_constant/{}/corr_results_{}_organelle_precent_pixels_0.0_in_image.csv".format(model['model'],organelle1,organelle2)                                        
                                        else:
                                            csv_file = "/sise/home/lionb/{}/predictions/{}/predictions_results_{}_in_image.csv".format(model['model'],organelle1,organelle2)
                                        if not os.path.exists(csv_file): 
                                            organelle1 = organelle
                                            organelle2 = "{}_{}".format(organelle,None)       
                                            if mode == "correlations":                    
                                                csv_file = "/sise/home/lionb/{}/predictions_correlations_constant/{}/corr_results_{}_organelle_precent_pixels_0.0_in_image.csv".format(model['model'],organelle1,organelle2)                                        
                                            else:
                                                csv_file = "/sise/home/lionb/{}/predictions/{}/predictions_results_{}_in_image.csv".format(model['model'],organelle1,organelle2)
                                            if not os.path.exists(csv_file): 
                                                organelle2 = "{}".format(organelle.replace(" ", "-"))       
                                                if mode == "correlations":                    
                                                    csv_file = "/sise/home/lionb/{}/predictions_correlations_constant/{}/corr_results_{}_organelle_precent_pixels_0.0_in_image.csv".format(model['model'],organelle1,organelle2)                                        
                                                else:
                                                    csv_file = "/sise/home/lionb/{}/predictions/{}/predictions_results_{}_in_image.csv".format(model['model'],organelle1,organelle2)
                                                if not os.path.exists(csv_file):
                                                    print("Check this...", csv_file)
                            else:
                                continue
                        if mode == "correlations":
                            csv_file = "/sise/home/lionb/{}/predictions_correlations_constant/{}/corr_results_{}_organelle_precent_pixels_0.0_in_image.csv".format(model['model'],organelle1,organelle2)
                        else:
                            csv_file = "/sise/home/lionb/{}/predictions/{}/predictions_results_{}_in_image.csv".format(model['model'],organelle1,organelle2)
                            
                        if os.path.exists(csv_file):
                            df = pd.read_csv(csv_file)
                            if v == "Vehicle" and drug is not None:
                                key = "{}_{}".format(model['organelle'],organelle)
                                if key in v_df.keys():
                                    v_df[key]['data'] = pd.concat([v_df[key]['data'],df])
                                else:
                                    v_df[key] = {}
                                    v_df[key]['data'] = df
                                    v_df[key]['target'] = model['organelle']
                                    v_df[key]['source'] = organelle
                                    v_df[key]['location'] = organelle_item['location']
                                    
                            else:
                                if method == "mean":
                                    column_averages = calculate_column_averages(df)
                                    if mode == "correlations":
                                        p_values = calculate_p_values(df, column_pairs)

                                        if not headers:
                                            headers = ['target','source','location_in_cell'] + list(column_averages.keys()) + [f'P-Value {col1} vs {col2}' for col1, col2 in column_pairs]

                                        row_data = list(column_averages.values()) + [p_values.get(f'P-Value ({col1}, {col2})', '') for col1, col2 in column_pairs]
                                    else:
                                        if not headers:
                                            headers = ['target','source','location_in_cell'] + list(column_averages.keys())

                                        row_data = list(column_averages.values())
                                                                        
                                    row_data.insert(0, model['organelle'])
                                    row_data.insert(1, organelle)
                                    row_data.insert(2, organelle_item['location'])
                                    output_data.append(row_data)
                                else:
                                    df['target'] = model['organelle']
                                    df['source'] = organelle
                                    df['location_in_cell'] = organelle_item['location']
                                    main_df = pd.concat([main_df,df])
                        else:
                            print("{} not exists.".format(csv_file))
                        
                except Exception as e:
                    print(str(e))
        if (len(main_df) or len(output_data) > 0) and v != "Vehicle":
            # Write averages and p-values to main file
            if mode == "correlations":
                output_file = "/sise/home/lionb/predictions_correlations_constant_{}_{}_main_organelle_precent_pixels_0.0_in_image".format(drug,v)
            else:
                output_file = "/sise/home/lionb/predictions_{}_{}_main_in_image".format(drug,v)
            df_output = pd.DataFrame(output_data, columns=headers).fillna(-1.0)
            if method == "mean":
                write_averages_to_csv(df_output, output_file, drug, v)
            else:
                write_all_to_csv(main_df, output_file, drug, v)
            
    if (len(v_df) > 0 or len(output_data) == 0) and v == "Vehicle":
        if len(v_df) > 0:
            for o_df in v_df.values():
                df = o_df['data']   
                if method == "mean":        
                    # Write averages and p-values to main file
                    column_averages = calculate_column_averages(df)
                    if mode == "correlations":
                        p_values = calculate_p_values(df, column_pairs)

                        if not headers:
                            headers = ['target','source','location_in_cell'] + list(column_averages.keys()) + [f'P-Value {col1} vs {col2}' for col1, col2 in column_pairs]

                        row_data = list(column_averages.values()) + [p_values.get(f'P-Value ({col1}, {col2})', '') for col1, col2 in column_pairs]
                    else:
                        if not headers:
                            headers = ['target','source','location_in_cell'] + list(column_averages.keys())

                        row_data = list(column_averages.values())                
                    row_data.insert(0, o_df['target'])
                    row_data.insert(1, o_df['source'])
                    row_data.insert(2, o_df['location'])
                    output_data.append(row_data)      
                else:
                    df['target'] = o_df['target']
                    df['source'] = o_df['source']
                    df['location_in_cell'] = o_df['location']
                    v_main_df = pd.concat([v_main_df,df])  
            if mode == "correlations":  
                output_file = "/sise/home/lionb/predictions_correlations_constant_{}_{}_main_organelle_precent_pixels_0.0_in_image".format(drug,v)
            else:
                output_file = "/sise/home/lionb/predictions_{}_{}_main_in_image".format(drug,v)
            df_output = pd.DataFrame(output_data, columns=headers).fillna(-1.0)
            if method == "mean":   
                write_averages_to_csv(df_output, output_file, None, "Vehicle")
            else:
                write_all_to_csv(v_main_df, output_file, None, "Vehicle")
        
if method == "mean":       
    # Create the bar chart with legend
    plt.figure(figsize=(20, 6))
    sns.barplot(y='mean_pcc', x='target', hue='drug', data=mean_df, palette='tab20')
    # plt.errorbar(x=mean_df['target'], y=mean_df['mean_pcc'], yerr=mean_df['pcc_std'], fmt='none', ecolor='black', capsize=5, errwidth=1)
    plt.ylabel('mean PCC', fontsize=12)
    plt.xlabel('Target', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title('mean PCC for Each Target', fontsize=14)
    plt.legend(title='Compounds', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Save the plot as an image file (e.g., PNG)
    if mode == "correlations":
        plt.savefig('bar_chart_means_with_legend_prediction vs noisy_prediction.png', bbox_inches='tight')
    else:
        plt.savefig('bar_chart_means_with_legend_prediction_vs_gt.png', bbox_inches='tight')



