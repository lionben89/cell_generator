# Re-import necessary libraries
import tifffile
import numpy as np
import matplotlib.pyplot as plt

# Define file path for confidence map
confidence_path = "/sise/home/lionb/mg_model_mito_13_05_24_1.5/spatial_pcc/27/prediction_to_noisy_spatial_pcc_27.tiff"

# Load confidence map
confidence_map = tifffile.imread(confidence_path)

# Select slice 19 (index 18 in zero-based indexing)
slice_idx = 18
confidence_slice = confidence_map[slice_idx]

# Plot histogram of confidence values
plt.figure(figsize=(6, 4))
plt.hist(confidence_slice.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel("Confidence Value")
plt.ylabel("Pixel Count")
plt.title("Histogram of Confidence Map (Slice 19)")
plt.grid(True)

# Show the histogram
plt.savefig("/sise/home/lionb/mg_model_mito_13_05_24_1.5/spatial_pcc/27/figure.svg")
plt.show()
