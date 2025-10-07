import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from dataset import DataGen
import global_vars as gv
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from utils import *
import tempfile
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import umap
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

load = True
mode = "_perturbation" #""
reducer = lambda histograms, n_components: perform_umap(histograms, n_components)
n_components = 4
n_clusters = 4

base_dir = "/sise/assafzar-group/assafzar/full_cells_fovs{}".format(mode)
gv.input = "channel_signal"
gv.target = "channel_target"
weighted_pcc = False


# Create a list of organelles from the parameters
params = [ 
          {"organelle":"Endoplasmic reticulum","unet":"../unet_model_22_05_22_er_128","target_col":"channel_target","collect_on":"Endoplasmic reticulum","mg":"../mg_model_er_13_05_24_1.5"},
          {"organelle":"Golgi","unet":"../unet_model_22_05_22_golgi_128","target_col":"channel_target","collect_on":'Golgi',"mg":"../mg_model_golgi_13_05_24_1.5"},
          {"organelle":"Actomyosin bundles","unet":"../unet_model_22_05_22_bundles_128","target_col":"channel_target","collect_on":'Actomyosin bundles',"mg":"../mg_model_bundles_13_05_24_1.0"},
          {"organelle":"Microtubules","unet":"../unet_model_22_05_22_microtubules_128","target_col":"channel_target","collect_on":'Microtubules',"mg":"../mg_model_microtubules_13_05_24_1.5"},
          {"organelle":"Actin filaments","unet":"../unet_model_22_05_22_actin_128","target_col":"channel_target","collect_on":'Actin filaments',"mg":"../mg_model_actin_13_05_24_1.5"},
          {"organelle":"DNA", "target_col":"channel_dna", "collect_on":None,"unet":"../unet_model_22_05_22_dna_128","mg":"../mg_model_dna_13_05_24_1.5b"},
          {"organelle":"Plasma-membrane", "target_col":"channel_membrane", "collect_on":None, "unet":"../unet_model_22_05_22_membrane_128","mg":"../mg_model_membrane_13_05_24_1.5"},
         ]

# Silhouette score to find the optimal number of clusters
def silhouette_analysis(embeddings, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(score)
    
    # # Plotting Silhouette scores
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o', color='blue')
    # plt.title('Silhouette Analysis for Optimal Number of Clusters')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Silhouette Score')
    # plt.show()

    return silhouette_scores

# Function to perform KMeans clustering
def perform_kmeans_clustering(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

# Function to load and preprocess images, and calculate histograms
def process_image_and_get_histogram(dataset, image_index):
    try:
        # Process image and get the histogram
        [target_image] = preprocess_image(dataset, int(image_index), [dataset.target_col], normalize=[False])
        # Flatten the target image and calculate its histogram
        hist, _ = np.histogram(target_image.ravel(), bins=4096,range=(0,2**16))
        return image_index, hist / np.sum(hist)  # Return image index to maintain order
    except Exception as e:
        print(f"Error processing image at index {image_index}: {e}")
        return image_index, None  # Return None if there was an error
    
# Function to load images and get histograms using threads
def get_histograms(dataset):
    histograms = [None] * len(dataset.df.data)  # Initialize the list with None values
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_image_and_get_histogram, dataset, idx): idx for idx in tqdm(dataset.df.data.index)}
        
        # Collect results in order of the image indices
        for future in tqdm(as_completed(futures)):
            idx, result = future.result()
            if result is not None:
                histograms[idx] = result  # Store the histogram in the correct order

    # Filter out any None values and return the histograms
    return np.array([hist for hist in histograms if hist is not None])

# Function to perform PCA on histograms
def perform_pca(histograms, n_components=2):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(histograms)
    return transformed

def perform_umap(histograms, n_components=2):
    umap_model = umap.UMAP(n_components=n_components,n_neighbors=50,min_dist=0.0)
    transformed = umap_model.fit_transform(histograms)
    return transformed

for param in params:
    gv.target = param["target_col"]
    ds_path = "/sise/assafzar-group/assafzar/full_cells_fovs{}/train_test_list/unet_predictions/metadata_with_efficacy_scores_and_unet_scores.csv".format(mode)
    new_ds_path = "/sise/assafzar-group/assafzar/full_cells_fovs{}/train_test_list/unet_predictions/metadata_with_efficacy_scores_and_unet_scores_embeddings_{}.csv".format(mode,param["organelle"])
    print("dataset:",param["organelle"])
    save_path = "/sise/assafzar-group/assafzar/full_cells_fovs{}/train_test_list/unet_predictions/histos_{}.npy".format(mode,param["organelle"])
    if not load:
        with tempfile.NamedTemporaryFile(mode='w', newline='') as temp_file:
            temp_file_path = temp_file.name  # Get the path to the temporary file
            temp_df = pd.read_csv(ds_path)
            if param["collect_on"] is not None:
                temp_df = temp_df[temp_df["Structure"]==param["collect_on"]]
            temp_df = temp_df.dropna(how="any",subset=[param["unet"],param["mg"]])
            temp_df.to_csv(temp_file_path, index=False)
            dataset = DataGen(temp_file_path ,gv.input,gv.target,image_path_col='combined_image_storage_path',batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
            print("# images in dataset:",dataset.df.data.shape[0])
            
            histograms = get_histograms(dataset)
            
            np.save(save_path,np.array(histograms))
    else:
        temp_df = pd.read_csv(new_ds_path)
        histograms = list(np.load(save_path))        
        
    embeddings = reducer(histograms, n_components=n_components)
    s_scores = silhouette_analysis(embeddings, max_clusters=10)
    print(s_scores)
    n_clusters = np.argmax(s_scores)+2
    print(n_clusters)
    clusters = perform_kmeans_clustering(embeddings, n_clusters=n_clusters)
    
    # Concatenate the PCA components with `param["mg"]`
    columns = []
    for i in range(n_components):
        columns.append("comp_{}".format(i))
    embeddings_df = pd.DataFrame(embeddings, columns=columns)
    embeddings_df[param["mg"]] = temp_df[param["mg"]].values - temp_df[param["mg"]].quantile(0.05) # Add mg values to the dataframe
    embeddings_df[param["mg"]] = embeddings_df[param["mg"]]/embeddings_df[param["mg"]].quantile(0.95)
    # Perform PCA again with the new dataframe (pca components + mg)
    embeddings_2 = reducer(embeddings_df.values, n_components=2)

    # Add PCA components to the DataFrame and save the updated file
    for i in range(2):
        temp_df[f"embeddings2_{i}_{param['organelle']}"] = embeddings_2[:, i]
    for i in range(n_components):
        temp_df[f"embeddings1_{i}_{param['organelle']}"] = embeddings[:, i]        
    temp_df['cluster'] = clusters
    temp_df.to_csv(new_ds_path, index=False)        
    # Normalize the unet column for color mapping
    norm = Normalize(vmin=temp_df[param["unet"]].quantile(0.05), vmax=temp_df[param["unet"]].quantile(0.95))
    cmap = cm.viridis  # You can change to another colormap if needed

    # Create a 1x3 grid for subplots
    fig, ax = plt.subplots(1, 4, figsize=(48, 12))  # Adjusted to 3 columns

    # First plot: Scatter of PCA colored by 'unet'
    sc1 = ax[0].scatter(embeddings[:, 0], embeddings[:, 1], c=temp_df[param["unet"]], cmap=cmap, norm=norm, s=50, edgecolor='k')
    cbar1 = plt.colorbar(sc1, ax=ax[0])
    cbar1.set_label('UNET Value', fontsize=12)
    ax[0].set_title(f"{param['organelle']} (colored by UNET)", fontsize=16)
    ax[0].set_xlabel('comp 1', fontsize=12)
    ax[0].set_ylabel('comp 2', fontsize=12)

    # Second plot: Scatter of PCA colored by 'mg'
    norm = Normalize(vmin=temp_df[param["mg"]].quantile(0.05), vmax=temp_df[param["mg"]].quantile(0.95))
    sc2 = ax[1].scatter(embeddings[:, 0], embeddings[:, 1], c=temp_df[param["mg"]], cmap=cmap, norm=norm, s=50, edgecolor='k')
    cbar2 = plt.colorbar(sc2, ax=ax[1])
    cbar2.set_label(param["mg"], fontsize=12)
    ax[1].set_title(f"{param['organelle']} (colored by {param['mg']})", fontsize=16)
    ax[1].set_xlabel('comp 1', fontsize=12)
    ax[1].set_ylabel('comp 2', fontsize=12)

    # Third plot: Scatter of re-PCA (colored by 'unet')
    norm = Normalize(vmin=temp_df[param["unet"]].quantile(0.05), vmax=temp_df[param["unet"]].quantile(0.95))
    sc3 = ax[2].scatter(embeddings_2[:, 0], embeddings_2[:, 1], c=temp_df[param["unet"]], cmap=cmap, norm=norm, s=50, edgecolor='k')
    cbar3 = plt.colorbar(sc3, ax=ax[2])
    cbar3.set_label('UNET Value', fontsize=12)
    ax[2].set_title(f"embeddings of {param['organelle']} after re-PCA (colored by UNET)", fontsize=16)
    ax[2].set_xlabel('comp 1', fontsize=12)
    ax[2].set_ylabel('comp 2', fontsize=12)
    
    # Fourth plot: Scatter of UMAP embeddings colored by KMeans clusters
    n_clusters = len(np.unique(clusters))  # Get number of clusters
    sc4 = ax[3].scatter(embeddings[:, 0], embeddings[:, 1], c=clusters, cmap='tab10', s=50, edgecolor='k')
    # cbar4 = plt.colorbar(sc4, ax=ax[3])
    # cbar4.set_label('Cluster', fontsize=12)
    ax[3].set_title(f"UMAP of {param['organelle']} (KMeans Clusters)", fontsize=16)
    ax[3].set_xlabel('comp 1', fontsize=12)
    ax[3].set_ylabel('comp 2', fontsize=12)

    # Manually add legend for KMeans clusters
    cluster_handles = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', markerfacecolor=cm.tab10(i), markersize=10) for i in range(n_clusters)]
    ax[3].legend(handles=cluster_handles, loc='best')  # Add the handles for each cluster to the legend

    
    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(f"/sise/home/lionb/figures/embeddings_target_{param['organelle']}{mode}.png")
    plt.show()  # Display the plot

