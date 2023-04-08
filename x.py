import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Set the file paths for the TIFF images
for i in [0,1,2,3,4,5]:
    index=i
    print(index)
    # unet_model_path = "/sise/home/lionb/unet_model_22_05_22_membrane_128"
    unet_model_path = "/sise/home/lionb/unet_model_22_05_22_actin_128"
    image1_path = "{}/predictions/{}/target_patch_{}.tiff".format(unet_model_path,index,index)
    image2_path = "{}/predictions/{}/prediction_patch_{}.tiff".format(unet_model_path,index,index)

    from cell_imaging_utils.image.image_utils import ImageUtils

    # Load the TIFF images into numpy arrays
    image1 = io.imread(image1_path)
    image2 = io.imread(image2_path)

    # Convert the images to matrices of pixel values
    matrix1 = image1.astype(float)
    matrix2 = image2.astype(float)

    ps=128
    s=16

    # Compute the Pearson correlation coefficient matrix for each slice of the 3D images
    corr_matrix = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))
    occur_matrix = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))
    for z in range(image1.shape[0]):
        for i in range(0,image1.shape[1] - ps + 1,s):
            for j in range(0,image1.shape[2] - ps + 1,s):
                corr_matrix[z,i:i+ps, j:j+ps] += np.corrcoef(matrix1[z,i:i+ps, j:j+ps].flatten(), matrix2[z,i:i+ps, j:j+ps].flatten())[0, 1]
                occur_matrix[z,i:i+ps, j:j+ps] +=1.0
                
    ImageUtils.imsave((corr_matrix/occur_matrix).astype(np.float16),"{}/predictions/{}/pm.tiff".format(unet_model_path,index))