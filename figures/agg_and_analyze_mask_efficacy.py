import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import mannwhitneyu

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

    # List of columns to plot unique values for
    columns_to_plot = ['WellName', 'Workflow', 'ColonyPosition', 'Passage', 
                       'InstrumentId', 'CellLine', 'CellPopulationId', 
                       'Clone', 'DataSetId', 'PlateId']

    # Calculate the number of unique values for each column
    unique_values_counts = {column: metadata_with_efficacy_scores_df[column].nunique() for column in columns_to_plot}

    # Plot unique values for each column
    plt.figure(figsize=(12, 8))
    plt.bar(unique_values_counts.keys(), unique_values_counts.values())
    plt.xlabel('Column Name')
    plt.ylabel('Number of Unique Values')
    plt.title('Number of Unique Values for Each Column')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('/sise/home/lionb/figures/unique_values_plot.png')
    plt.show()

    # Print unique values for each column
    for column in columns_to_plot:
        unique_values = metadata_with_efficacy_scores_df[column].unique()
        print(f"\nUnique values in {column}:")
        print(unique_values)

# List of columns with the results to plot
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

# Function to plot box plots for each column and result
def plot_box_plots(data, columns, params):
    for param in tqdm(params):
        ## Collect training and testing data for the model
        control_df = pd.read_csv("/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/{}/image_list_with_metadata__with_efficacy_scores_full.csv".format(param["organelle"]))
        for x_col in columns:
            plt.figure(figsize=(30, 6))
            unique_values = np.sort(data[x_col].unique())
            groups = [data[data[x_col] == val][param["model"]].dropna() for val in unique_values]
            control_values = control_df[param["model"]].dropna()
            
            # Calculate the lower and upper bounds for the control group without outliers
            q1 = np.percentile(control_values, 25)
            q3 = np.percentile(control_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            control_values_no_outliers = control_values[(control_values >= lower_bound) & (control_values <= upper_bound)]
            lower_bound_no_outliers = np.min(control_values_no_outliers)
            upper_bound_no_outliers = np.max(control_values_no_outliers)
            
            # Add control values to the groups
            groups.append(control_values)
            labels = list(unique_values) + ['Train set + Test set']
            
            # Perform significance tests against the lower bound without outliers
            for i, group in enumerate(groups[:-1]):
                stat, p_value = mannwhitneyu(group, control_values_no_outliers, alternative='less')
                if p_value < 0.05 and np.median(group) < lower_bound_no_outliers:
                    plt.text(i + 1, np.median(group), '**', ha='center', va='bottom', color='blue', fontsize=14)
            
            y_col = param["model"][3:]
            plt.boxplot(groups, labels=labels)
            plt.axhline(lower_bound_no_outliers, color='green', linestyle='--')
            plt.axhline(upper_bound_no_outliers, color='green', linestyle='--')
            plt.ylim(0.5, max(upper_bound_no_outliers, max([group.max() for group in groups])) + 0.1)
            plt.title(f"{x_col} vs {y_col} mask efficacy")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.xticks(rotation=90, ha='right')
            plt.tight_layout()
            plt.savefig('/sise/home/lionb/figures/{}_vs_{}.png'.format(y_col, x_col))
            plt.close()

# Plot box plots for each specified column and result column
plot_box_plots(metadata_with_efficacy_scores_df, columns_to_plot, params)
