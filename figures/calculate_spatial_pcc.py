import tensorflow as tf
import tensorflow.keras as keras
from metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
import global_vars as gv
from utils import *
import os
import cv2
from dataset import DataGen


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

gv.input = "channel_signal"
gv.target = "channel_target"
weighted_pcc = False


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


## Patch steps
xy_step = 64
z_step = 16

batch_size = 4

def calculate_spatial_pcc(image1,image2):
    # Compute the Pearson correlation coefficient matrix for each slice of the 3D images
    corr_matrix = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))
    occur_matrix = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))
    for z in range(image1.shape[0]):
        for i in range(0,image1.shape[1] - xy_step + 1,z_step):
            for j in range(0,image1.shape[2] - xy_step + 1,z_step):
                corr_matrix[z,i:i+xy_step, j:j+xy_step] += np.corrcoef(image1[z,i:i+xy_step, j:j+xy_step].flatten(), image2[z,i:i+xy_step, j:j+xy_step].flatten())[0, 1]
                occur_matrix[z,i:i+xy_step, j:j+xy_step] +=1.0
    return (corr_matrix/occur_matrix).astype(np.float16)           

def predict_images_and_calculate_spatial_pcc(dataset,model_path=gv.model_path,model=None,images=range(10),weighted_pcc=False,noise_scale=1.5):
    """This method run analysis that find the std of the noise that need to be used for the input data
        We test the prediction of noisy input by adding noise to the input with different stds, the noisy input is 0.5 signal, 0.5 noise.
    Args:
        dataset (_type_): _description_
        model_path (_type_, optional): _description_. Defaults to gv.model_path.
        model (_type_, optional): _description_. Defaults to None.
        images (_type_, optional): _description_. Defaults to range(10).
        weighted_pcc (bool, optional): _description_. Defaults to False.
        #######TODO: need to change docs
    """
    batch_size=4
    ##Load model
    if model is None:
        print("Loading model:",model_path)
        model = keras.models.load_model(model_path)
        
    dir_path = "{}/spatial_pcc".format(model_path)
    create_dir_if_not_exist(dir_path)

    count=0
    for image_index in images:
        count+=1
        print("image index: {}/{}".format(count,len(images)))
        
        slice_by = None
        print("preprocess..")
        input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image = preprocess_image(dataset,int(image_index),[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane","membrane_seg"],normalize=[True,True,False,True,True,False],slice_by=slice_by)
        pz_end = input_image.shape[0]
        px_start=0
        py_start=0
        pz_start = 0
        px_end=input_image.shape[1]
        py_end=input_image.shape[2]
        if target_seg_image is not None:
            target_seg_image = target_seg_image/255.
        else:
            target_seg_image = np.zeros_like(target_image)
            
        if mem_seg_image is not None:
            mem_seg_image = mem_seg_image/255.
        else:
            mem_seg_image = np.ones_like(target_image)
        
        ## Collect patchs
        input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,input_image, gv.patch_size, xy_step, z_step)
        if weighted_pcc:
            target_seg_image_dilated = np.copy(target_seg_image)
            for h in range(target_seg_image.shape[1]):
                target_seg_image_dilated[0, h, :, :] = cv2.dilate(target_seg_image_dilated[0, h, :, :].astype(np.uint8), np.ones((25,25)))  
        else:
            target_seg_image_dilated = None
        
        weights = get_weights(input_patchs[0].shape)    
        
        ## Batch predict
        ## Predicte unet and mask
        
        print("batch predict...")
        unet_patchs_p = predict(model.unet,input_patchs,batch_size,gv.patch_size)
        mask_patchs_p = predict(model,input_patchs,batch_size,gv.patch_size)
        
        ## Back to image 
        unet_p,mask_p_full,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,mask_patchs_p,np.ones_like(mask_patchs_p)],weights,input_image.shape,gv.patch_size,xy_step,z_step)
        del unet_patchs_p
        mask_p = mask_p_full/d
        
        ## Create noise vector
        normal_noise = tf.random.normal(tf.shape(input_patchs),stddev=noise_scale,dtype=tf.float64)
        ## Create noisy input and predict unet
        mask_patchs_p_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p, gv.patch_size, xy_step, z_step)
        mask_noise_patchs = (normal_noise*(1-mask_patchs_p_term))
        input_patchs_p = (mask_patchs_p_term*input_patchs)+mask_noise_patchs
        unet_noise_patchs_p = predict(model.unet,input_patchs_p.numpy(),batch_size,gv.patch_size)

        ## Back to image 
        input_p,unet_noise_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p],weights,input_image.shape,gv.patch_size,xy_step,z_step)
        ImageUtils.imsave((unet_p/d).astype(np.float16),"{}/unet_prediction_{}.tiff".format(dir_path,image_index)) 
        ImageUtils.imsave((unet_noise_p/d).astype(np.float16),"{}/unet_noise_prediction_{}.tiff".format(dir_path,image_index)) 
        
        del input_patchs_p      
        del unet_noise_patchs_p
        del input_p
        del unet_noise_p
        del normal_noise
        del input_patchs
        del mask_patchs_p_term
        del mask_noise_patchs
        del mask_patchs_p_term
        
        prediction_to_gt_spatial_pcc = calculate_spatial_pcc(target_image,unet_p/d)
        ImageUtils.imsave(prediction_to_gt_spatial_pcc,"{}/prediction_to_gt_spatial_pcc_{}.tiff".format(dir_path,image_index))
        
        prediction_to_noisy_spatial_pcc = calculate_spatial_pcc(unet_noise_p/d,unet_p/d)
        ImageUtils.imsave(prediction_to_noisy_spatial_pcc,"{}/prediction_to_noisy_spatial_pcc_{}.tiff".format(dir_path,image_index))

        

for param in params:
    try:
        print(param["model"])
        base_path = "/sise/assafzar-group/assafzar/full_cells_fovs/train_test_list/spatial_pcc/{}".format(param["model"].split('/')[-1])
        create_dir_if_not_exist(base_path)
        ds_path = "{}/image_list.csv".format(base_path)
        dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
        print("# images in dataset:",dataset.df.data.shape[0])  
        predict_images_and_calculate_spatial_pcc(dataset,model_path=param["model"],images=range(min(10,dataset.df.data.shape[0])),weighted_pcc=False,noise_scale=param["noise"])            
    except Exception as e:
        print(e)
