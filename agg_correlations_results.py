import pandas as pd
import os
from scipy import stats

def calculate_column_averages(csv_file):
    df = pd.read_csv(csv_file)
    if 'importance_in_organelle' in df.columns:
        df = df.drop('importance_in_organelle',axis=1)
    df_mean = df.iloc[:, 1:].mean()

    column_averages = {}

    for column, average in df_mean.items():
        column_averages[column] = average

    return column_averages

def calculate_p_values(csv_file, column_pairs):
    df = pd.read_csv(csv_file)
    df = df.iloc[:, 1:]
    p_values = {}

    for column_pair in column_pairs:
        col1, col2 = column_pair
        p_value = stats.ttest_ind(df[col1], df[col2],alternative='less').pvalue
        p_values[f'P-Value ({col1}, {col2})'] = p_value

    return p_values

def write_averages_to_csv(output_data, headers, output_file):
    df_output = pd.DataFrame(output_data, columns=headers)
    df_output.to_csv(output_file, index=False)

# Define pairs of columns for p-value calculation
column_pairs = [('pcc_wo_organelle_pixels', 'pcc_wo_organelle_pixels_wo_mask'),('pcc_wo_organelle_pixels','pcc_random_flip'),('pcc_wo_organelle_pixels_wo_mask','pcc_random_flip'),('pcc_wo_organelle_pixels','pcc_random_pixels'),('pcc_wo_organelle_pixels_wo_mask','pcc_random_pixels')]  # Update with your desired column pairs

models = [{'model':'mg_model_microtubules_10_06_22_5_0_new','organelle':"Microtubules"},{'model':'mg_model_golgi_10_06_22_5_0_new','organelle':"Golgi"},{'model':'mg_model_actin_10_06_22_5_0_new','organelle':"Actin-filaments"},{'model':'mg_model_membrane_10_06_22_5_0_new','organelle':"Plasma-membrane"},{'model':'mg_model_ne_10_06_22_5_0_new','organelle':"Nuclear-envelope"},{'model':'mg_model_ngc_10_06_22_5_0_new','organelle':"Nucleolus-(Granular-Component)"},{'model':'mg_model_mito_10_06_22_5_0_new','organelle':"Mitochondria"}]
organelles=[{'name':"Nuclear-envelope",'location':"Nuclear periphry"},{'name':"Endoplasmic-reticulum",'location':"Cytoplasm"},{'name':"Golgi",'location':"Cytoplasm"},{'name':"Plasma-membrane",'location':"Cell periphery"},{'name':"Nucleolus-(Dense-Fibrillar-Component)",'location':"Nuclear"},{'name':"Nucleolus-(Granular-Component)",'location':"Nuclear"},{'name':"Microtubules",'location':"Cytoplasm"}\
    ,{'name':"Tight-junctions",'location':"Apical domain"},{'name':"Mitochondria",'location':"Cytoplasm"},{'name':"Actomyosin-bundles",'location':"Cell periphery"},{'name':"Actin-filaments",'location':"Cell periphery"},{'name':"Desmosomes",'location':"Apical domain"},{'name':"Lysosome",'location':"Cytoplasm"},{'name':"Adherens-junctions",'location':"Apical domain"},{'name':"Gap-junctions",'location':"Apical domain"},{'name':"Matrix-adhesions",'location':"Basal domain"},{'name':"Peroxisomes",'location':"Cytoplasm"},{'name':"Endosomes",'location':"Cytoplasm"}]

drugs = ["s-Nitro-Blebbistatin","Rapamycin","Paclitaxol","Staurosporine","Brefeldin",None]

for drug in drugs:
    for v in [None,"Vehicle"]:
        # Process each CSV file and calculate column averages and p-values
        output_data = []
        headers = []
        for model in models:
            for organelle_item in organelles:
                try:
                    organelle = organelle_item['name']
                    if drug is not None:
                        organelle1 = "{}_{}".format(organelle.replace("-", " "),drug)
                        if v is None:
                            organelle2 = organelle1
                        else:
                            organelle2 = "{}_{}".format(organelle.replace("-", " "),v)
                    else:
                        if v is None:
                            organelle1 = organelle
                            organelle2 = organelle       
                        else:
                            continue
                    csv_file = "/sise/home/lionb/{}/predictions_correlations_constant/{}/corr_results_{}_organelle_precent_pixels_0.001_in_image.csv".format(model['model'],organelle1,organelle2)
                    if os.path.exists(csv_file):
                        column_averages = calculate_column_averages(csv_file)
                        p_values = calculate_p_values(csv_file, column_pairs)

                        if not headers:
                            headers = ['target','source','location_in_cell'] + list(column_averages.keys()) + [f'P-Value {col1} vs {col2}' for col1, col2 in column_pairs]

                        row_data = list(column_averages.values()) + [p_values.get(f'P-Value ({col1}, {col2})', '') for col1, col2 in column_pairs]
                        row_data.insert(0, model['organelle'])
                        row_data.insert(1, organelle)
                        row_data.insert(2, organelle_item['location'])
                        output_data.append(row_data)
                    else:
                        print("{} not exists.".format(csv_file))
                        
                except Exception as e:
                    print(str(e))
        if len(output_data) > 0:
            # Write averages and p-values to main file
            output_file = "/sise/home/lionb/predictions_correlations_constant_{}_{}_main_organelle_precent_pixels_0.001_in_image.csv".format(drug,v)
            write_averages_to_csv(output_data, headers, output_file)



