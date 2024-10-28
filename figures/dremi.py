import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity, NearestNeighbors
import warnings
import numbers
from figure_config import figure_config

def dermi_kde(x, y, x_col="Feature1",y_col="Feature2", n_bins=20, n_mesh=3, plot=True,title=""):
    """Compute Density Resampled Estimate of Mutual Information using KDE."""
    x, y = _vector_coerce_two_dense(x, y)

    if np.count_nonzero(x - x[0]) == 0 or np.count_nonzero(y - y[0]) == 0:
        warnings.warn(
            "Attempting to calculate DREMI on a constant array. Returning `0`",
            UserWarning,
        )
        return 0

    # Z-score X and Y
    x = np.clip(stats.zscore(x),-5,5)
    y = np.clip(stats.zscore(y),-5,5)

    # Create bin and mesh points
    x_bins = np.linspace(min(x), max(x), n_bins + 1)
    y_bins = np.linspace(min(y), max(y), n_bins + 1)
    x_mesh = np.linspace(min(x), max(x), ((n_mesh + 1) * n_bins) + 1)
    y_mesh = np.linspace(min(y), max(y), ((n_mesh + 1) * n_bins) + 1)

    # Perform joint density estimation using KDE
    data_subset = np.vstack([x, y])
    kde_joint = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_subset.T)

    # Evaluate joint density on the mesh points
    mesh_points = np.vstack(
        [np.tile(x_mesh, len(y_mesh)), np.repeat(y_mesh, len(x_mesh))]
    ).T
    log_joint_density = kde_joint.score_samples(mesh_points)
    joint_density = np.exp(log_joint_density)

    # Sum the densities of each point over the bins
    bin_density, _, _ = np.histogram2d(
        mesh_points[:, 0],
        mesh_points[:, 1],
        bins=[x_bins, y_bins],
        weights=joint_density,
    )
    bin_density = bin_density.T
    bin_density = bin_density / np.sum(bin_density)

    # Compute conditional density estimate of Y given X
    conditional_density = bin_density / np.sum(bin_density, axis=0)
    
    # Rescale each column by its maximum value
    column_max = np.max(conditional_density, axis=0)
    rescaled_density = conditional_density / column_max

    # Mutual information
    marginal_entropy = stats.entropy(np.sum(bin_density, axis=1), base=2)
    cond_entropies = stats.entropy(conditional_density, base=2)
    cond_sums = np.sum(bin_density, axis=0)
    conditional_entropy = np.sum(cond_entropies * cond_sums)
    mutual_info = marginal_entropy - conditional_entropy

    # DREMI
    marginal_entropy_norm = stats.entropy(np.sum(conditional_density, axis=1), base=2)
    cond_sums_norm = np.mean(conditional_density)
    conditional_entropy_norm = np.sum(cond_entropies * cond_sums_norm)
    dremi = marginal_entropy_norm - conditional_entropy_norm

    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    
        # Plot raw data
        axes[0].scatter(x, y, c="k", s=4)
        axes[0].set_title("Raw scores", fontsize=14, fontname=figure_config["font"])
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])
        axes[0].set_ylabel(y_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot KDE joint density
        n = ((n_mesh + 1) * n_bins) + 1
        axes[1].imshow(
            joint_density.reshape(n, n), cmap="inferno", origin="lower", aspect="auto"
        )
        for b in np.linspace(0, n, n_bins + 1):
            axes[1].axhline(b - 0.5, c="grey", linewidth=1)
        for b in np.linspace(0, n, n_bins + 1):
            axes[1].axvline(b - 0.5, c="grey", linewidth=1)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title("KDE Density", fontsize=14, fontname=figure_config["font"])
        axes[1].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot joint probability
        axes[2].imshow(bin_density, cmap="inferno", origin="lower", aspect="auto")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].set_title("Joint Prob.[MI={:.2f}]".format(mutual_info), fontsize=14, fontname=figure_config["font"])
        axes[2].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot conditional probability
        axes[3].imshow(rescaled_density, cmap="inferno", origin="lower", aspect="auto")
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].set_title("Conditional Prob.[DREMI={:.2f}]".format(dremi), fontsize=14, fontname=figure_config["font"])
        axes[3].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        fig.suptitle(title, fontsize=figure_config["organelle"],fontname=figure_config["font"],y=0.92)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("/sise/home/lionb/figures/kde_dermi_{}.png".format(title))

    return dremi

def dermi_knn(x, y, x_col="Feature1",y_col="Feature2",k=10, n_bins=20, n_mesh=3, n_jobs=1, plot=False,title=""):
    """Compute kNN conditional Density Resampled Estimate of Mutual Information."""
    x, y = _vector_coerce_two_dense(x, y)

    if np.count_nonzero(x - x[0]) == 0 or np.count_nonzero(y - y[0]) == 0:
        warnings.warn(
            "Attempting to calculate kNN-DREMI on a constant array. Returning `0`",
            UserWarning,
        )
        return -1

    if not isinstance(k, numbers.Integral):
        raise ValueError("Expected k as an integer. Got {}".format(type(k)))
    if not isinstance(n_bins, numbers.Integral):
        raise ValueError("Expected n_bins as an integer. Got {}".format(type(n_bins)))
    if not isinstance(n_mesh, numbers.Integral):
        raise ValueError("Expected n_mesh as an integer. Got {}".format(type(n_mesh)))

    # Z-score X and Y
    x = np.clip(stats.zscore(x),-5,5)
    y = np.clip(stats.zscore(y),-5,5)

    # Create bin and mesh points
    x_bins = np.linspace(min(x), max(x), n_bins + 1)
    y_bins = np.linspace(min(y), max(y), n_bins + 1)
    x_mesh = np.linspace(min(x), max(x), ((n_mesh + 1) * n_bins) + 1)
    y_mesh = np.linspace(min(y), max(y), ((n_mesh + 1) * n_bins) + 1)

    mesh_points = np.vstack(
        [np.tile(x_mesh, len(y_mesh)), np.repeat(y_mesh, len(x_mesh))]
    ).T

    knn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs).fit(np.vstack([x, y]).T)
    dists, _ = knn.kneighbors(mesh_points)

    area = np.pi * (dists[:, -1] ** 2)
    density = k / (area + 1e-10)  # Add small constant to avoid division by zero

    mesh_mask = np.logical_or(
        np.isin(mesh_points[:, 0], x_bins), np.isin(mesh_points[:, 1], y_bins)
    )
    bin_density, _, _ = np.histogram2d(
        mesh_points[~mesh_mask, 0],
        mesh_points[~mesh_mask, 1],
        bins=[x_bins, y_bins],
        weights=density[~mesh_mask],
    )
    bin_density = bin_density.T
    bin_density = bin_density / np.sum(bin_density)

    drevi = bin_density / np.sum(bin_density, axis=0)
    cond_entropies = stats.entropy(drevi, base=2)

    marginal_entropy = stats.entropy(np.sum(bin_density, axis=1), base=2)
    cond_sums = np.sum(bin_density, axis=0)
    conditional_entropy = np.sum(cond_entropies * cond_sums)
    mutual_info = marginal_entropy - conditional_entropy

    marginal_entropy_norm = stats.entropy(np.sum(drevi, axis=1), base=2)
    cond_sums_norm = np.mean(drevi)
    conditional_entropy_norm = np.sum(cond_entropies * cond_sums_norm)

    dremi = marginal_entropy_norm - conditional_entropy_norm

    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        
        # Plot raw data
        axes[0].scatter(x, y, c="k", s=4)
        axes[0].set_title("Row scores",fontsize=14, fontname=figure_config["font"])
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])
        axes[0].set_ylabel(y_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot kNN density
        n = ((n_mesh + 1) * n_bins) + 1
        axes[1].imshow(
            np.log(density.reshape(n, n)), cmap="inferno", origin="lower", aspect="auto"
        )
        for b in np.linspace(0, n, n_bins + 1):
            axes[1].axhline(b - 0.5, c="grey", linewidth=1)

        for b in np.linspace(0, n, n_bins + 1):
            axes[1].axvline(b - 0.5, c="grey", linewidth=1)

        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title("kNN Density", fontsize=14, fontname=figure_config["font"])
        axes[1].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot joint probability
        axes[2].imshow(bin_density, cmap="inferno", origin="lower", aspect="auto")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].set_title(
            "Joint Prob.[MI={:.2f}]".format(mutual_info), fontsize=14, fontname=figure_config["font"]
        )
        axes[2].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        # Plot conditional probability
        axes[3].imshow(drevi, cmap="inferno", origin="lower", aspect="auto")
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].set_title(
            "Conditional Prob.[DREMI={:.2f}]".format(dremi), fontsize=14, fontname=figure_config["font"]
        )
        axes[3].set_xlabel(x_col, fontsize=figure_config["axis"], fontname=figure_config["font"])

        fig.suptitle(title, fontsize=figure_config["organelle"], fontname=figure_config["font"],y=0.92)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("/sise/home/lionb/figures/knn_dermi_{}.png".format(title))

        return dremi

def _vector_coerce_two_dense(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    if len(x) != len(y):
        raise ValueError("Expected x and y to be the same length. Got {} and {}".format(len(x), len(y)))
    return x, y

# Generate sample data for testing
# np.random.seed(0)
# x = np.random.rand(100)
# y = np.exp(x)+x*0.5 + np.random.rand(100)

import pandas as pd

corr_df = pd.read_csv("/sise/home/lionb/cell_generator/figures/membrane_correlation.csv")
x_col = '../mg_model_membrane_13_05_24_1.5'
x = corr_df[x_col].values
y_col = '../unet_model_22_05_22_membrane_128'
y = corr_df[y_col].values
title = "Plasma-membrane"

# corr_df = pd.read_csv("/sise/home/lionb/cell_generator/figures/dna correlation.csv")
# corr_df = corr_df.dropna(how='any')
# x_col = '../mg_model_dna_13_05_24_1.5b'
# x = corr_df[x_col].values
# y_col = '../unet_model_22_05_22_dna_128'
# y = corr_df[y_col].values
# title = "DNA"

# Plot DREMI with KDE and kNN-DREMI
print("Plotting DREMI with KDE:")
dremi_value_kde = dermi_kde(x, y, "mask efficacy [PCC]", "UNET prediction vs GT [PCC]", n_bins=20, n_mesh=10, plot=True,title=title)

print("Plotting kNN-DREMI:")
dremi_value_knn = dermi_knn(x, y, "mask efficacy [PCC]", "UNET prediction vs GT [PCC]", k=12, n_bins=20, n_mesh=10, n_jobs=2, plot=True,title=title)

print("DREMI with KDE:", dremi_value_kde)
print("kNN-DREMI:", dremi_value_knn)