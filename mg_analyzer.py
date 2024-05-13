import gc
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
from metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
import global_vars as gv
from utils import *
import os
import matplotlib.pyplot as plt
import cv2

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

## Image center
center_xy = [312,462] #[200,100]
margin=[192,256] #[256,256]#

## Patch steps
xy_step = 64
z_step = 16

batch_size = 4
noise_scale = 5.0

def find_noise_scale(dataset,model_path=gv.model_path,model=None,images=range(10),weighted_pcc=False):
    
    batch_size=4
    
    ##Load model
    if model is None:
        print("Loading model:",model_path)
        model = keras.models.load_model(model_path)
        
    ## Create noise vars
    noise_start = 0.0
    noise_step = 0.5
    noise_stop = 4.5
    noises = np.arange(noise_start,noise_stop,noise_step)
        
    dir_path = "{}".format(model_path)
    create_dir_if_not_exist(dir_path)
    
    # results of pcc for the prediction from original and noisy input by noise std
    pcc_results = DatasetMetadataSCV("{}/noise_pcc_resuls.csv".format(dir_path))
    pcc_results.create_header(noises)

    count=0
    for image_index in images:
        count+=1
        print("image index: {}/{}".format(count,len(images)))
        pccs = []
        
        ## Preprocess images
        # px_start=center_xy[0]-margin[0]-xy_step
        # py_start=center_xy[1]-margin[1]-xy_step
        # pz_start = 0
        # px_end=center_xy[0]+margin[0]+xy_step
        # py_end=center_xy[1]+margin[1]+xy_step
        
        slice_by = None
        # slice_by = [None,(px_start,px_end),(py_start,py_end)]
        print("preprocess..")
        input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image = preprocess_image(dataset,int(image_index),[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane","membrane_seg"],normalize=[True,True,False,True,True,False],slice_by=slice_by)
        pz_end = input_image.shape[0]
        if slice_by is None:
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
        unet_patchs_p = predict(model,input_patchs,batch_size,gv.patch_size)
        
        ## Back to image 
        unet_p,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,np.ones_like(input_patchs)],weights,input_image.shape,gv.patch_size,xy_step,z_step)
        
        pcc_gt = pearson_corr((target_image), (unet_p/d),target_seg_image_dilated)
        print("pearson with gt corr for image:{} is :{}".format(image_index,pcc_gt))
        
        del unet_patchs_p
        
        for noise_scale in noises:
            print(noise_scale)
            ## Create noise vector
            normal_noise = tf.random.normal(tf.shape(input_patchs),stddev=noise_scale,dtype=tf.float64)
            
            ## Create noisy input and predict unet
            input_patchs_p = input_patchs+normal_noise
            unet_noise_patchs_p = predict(model,input_patchs_p.numpy(),batch_size,gv.patch_size)
    
            ## Back to image 
            input_p,unet_noise_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p],weights,input_image.shape,gv.patch_size,xy_step,z_step)
            
            pcc = pearson_corr((unet_p/d), (unet_noise_p/d),target_seg_image_dilated)
            
            pccs.append(pcc)
            print("pearson corr for image:{} with noise std:{}, is :{}".format(image_index,noise_scale,pcc))
            
            del input_patchs_p      
            del unet_noise_patchs_p
            del input_p
            del unet_noise_p
            del normal_noise
            
        pcc_results.add_row(pccs)
        
        del input_patchs
        
    pcc_results.create()

def analyze_th(dataset,mode,mask_image=None,manual_th="full",save_image=True,save_histo=False,weighted_pcc = False, model_path=gv.model_path,model=None):
    """method that generate masks and predictions by thresholding the importance mask with different TH

    Args:
        dataset (DataGen): instance that point to the dataset, pertrub or regular according to the organelle.
        mode (str): "agg" - th all the values below the th
                    "loo" - th all the values except the current th
                    "mask" - by external mask - used to evaluate gradcam, guided back propagation and manual validations 
                    "regular" - use the importance mask as is without binary th, if manual_th is not "full" then use the manual_th value
        mask_image (np ndarray, optional): if mode == "mask" then this is the mask to use. Defaults to None. values are 0 or 255
        manual_th (str, float optional): _description_. Defaults to "full".
        save_image (bool, int optional): if True save all images, if False save nothing, if int, save the first images according to the int. Defaults to True.
        save_histo (bool, optional): Save histograms of the importance masks. Defaults to False.
        weighted_pcc (bool, optional): use regular PCC or weighted_PCC. Defaults to False.
        model_path (str, optional): use to load model if not passed and dir to save results. Defaults to gv.model_path.
        model (model, optional): model if exists, if None method will load model from model_path. Defaults to None.
    """
    ##Load model
    if model is None:
        print("Loading model:",param["model"])
    model = keras.models.load_model(param["model"])
    ## Create thresholds
    num_bins = 10 # number of bins for the histogram
    ths_start = 0.1
    ths_step = 0.1
    ths_stop = 1.1
    ths = np.arange(ths_start,ths_stop,ths_step)
    ## Create main dir
    if mode=="agg":
        pred_path = "predictions_agg"
    elif mode=="loo":
        pred_path = "predictions_loo"
    elif mode=="mask":
        pred_path = "predictions_masked/gc"  
        # ths = [0.0]
        ths=[0.0,0.0000125,0.000025,0.00005,0.000075,0.0001,0.00015,0.0002,0.00025,0.0003,0.0004,0.008]
    elif mode=="regular":
        pred_path="predictions"
        ths=[manual_th]
    if compound is not None:
        pred_path = "{}_{}".format(pred_path,compound)
        
    dir_path = "{}/{}".format(model_path,pred_path)
    create_dir_if_not_exist(dir_path)
    
    # results of pcc for the prediction from original and noisy input by th
    pcc_results = DatasetMetadataSCV("{}/pcc_resuls.csv".format(dir_path))
    pcc_results.create_header(ths)
    
    # results of the size of the mask by th
    mask_results = DatasetMetadataSCV("{}/mask_size_resuls.csv".format(dir_path))
    mask_results.create_header(ths)
    
    #the % of the intersection of true organelle pixels and the mask pixels by th
    context_results = DatasetMetadataSCV("{}/context_resuls.csv".format(dir_path))
    context_results.create_header(ths)
    if save_histo:
        histos = DatasetMetadataSCV("{}/histograms.csv".format(dir_path))
        bins = np.linspace(0,1.0,num_bins)
        histos.create_header(bins)
        histos_l = int(np.ceil(np.sqrt(len(images)+1)))
        histos_fig, histos_axs = plt.subplots(histos_l, histos_l, sharey=True)
        histos_fig.set_figheight(40)
        histos_fig.set_figwidth(40)
        sum_histos = np.zeros_like(bins)
    count=0
    for image_index in images:
        print("image index: {}/{}".format(count+1,len(images)))
        pccs = []
        mask_sizes = []
        context=[]
        ## Create image dir
        if save_image:
            create_dir_if_not_exist("{}/{}".format(dir_path,image_index))
        
        ## Preprocess images
        px_start=center_xy[0]-margin[0]-xy_step
        py_start=center_xy[1]-margin[1]-xy_step
        pz_start = 0
        px_end=center_xy[0]+margin[0]+xy_step
        py_end=center_xy[1]+margin[1]+xy_step
        
        # slice_by = None
        slice_by = [None,(px_start,px_end),(py_start,py_end)]
        print("preprocess..")
        input_image,target_image,target_seg_image,nuc_image,mem_image,mem_seg_image = preprocess_image(dataset,int(image_index),[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane","membrane_seg"],normalize=[True,True,False,True,True,False],slice_by=slice_by)
        pz_end = input_image.shape[0]
        if slice_by is None:
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
            # mem_seg_image = np.ones_like(target_image)
            mem_seg_image = mem_seg_image/255.
        else:
            mem_seg_image = np.ones_like(target_image)
        
        weights = None
        
        ## Collect patchs
        input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,input_image, gv.patch_size, xy_step, z_step)
        if weighted_pcc:
            target_seg_image_dilated = np.copy(target_seg_image)
            for h in range(target_seg_image.shape[1]):
                target_seg_image_dilated[0, h, :, :] = cv2.dilate(target_seg_image_dilated[0, h, :, :].astype(np.uint8), np.ones((25,25)))  
        else:
            target_seg_image_dilated = None
            
        ## Batch predict
        ## Predicte unet and mask
        print("batch predict...")
        unet_patchs_p = predict(model.unet,input_patchs,batch_size,gv.patch_size)#model.unet.predict(tf.convert_to_tensor(input_patchs),batch_size=batch_size)
        mask_patchs_p = predict(model,input_patchs,batch_size,gv.patch_size)#model.predict(tf.convert_to_tensor(input_patchs),batch_size=batch_size)
        
        ## Create noise vector
        normal_noise = tf.random.normal(tf.shape(mask_patchs_p),stddev=noise_scale,dtype=tf.float64)
        # normal_noise = tf.random.uniform(tf.shape(mask_patchs_p),maxval=noise_scale,dtype=tf.float64)
        
        ## Back to image 
        weights = get_weights(input_patchs[0].shape)
        unet_p,mask_p_full,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,mask_patchs_p,np.ones_like(mask_patchs_p)],weights,input_image.shape,gv.patch_size,xy_step,z_step)
        # mem_seg_image = np.ones_like(unet_p)
        mask_p_full = mask_p_full
        
        for th in ths:
            print(th)
            print("mask predict ...")
            if save_image:
                create_dir_if_not_exist("{}/{}/{}".format(dir_path,image_index,th))
            if mode=="mask" and mask_image is not None:
                # mask_image_ndarray = target_seg_image
                
                mask_image_ndarray = ImageUtils.image_to_ndarray(ImageUtils.imread(mask_image))/255.
                mask_image_ndarray = np.expand_dims(mask_image_ndarray[0],axis=-1)
                mask_image_ndarray = slice_image(mask_image_ndarray,slice_by)
                
                ## mask_p = (1. - mask_image_ndarray*0.0)
                mask_p = tf.cast(tf.where(mask_image_ndarray>=th,1.0,0.5),tf.float64).numpy()
                mask_p_binary = tf.cast(tf.where(mask_image_ndarray>=th,1.0,0.0),tf.float64).numpy()
            elif mode=="agg":
                mask_p_binary = tf.cast(tf.where((mask_p_full/d)>th,1.0,0.0),tf.float64).numpy()
                mask_p = tf.cast(tf.where((mask_p_full/d)>th,1.0,0.5),tf.float64).numpy()
            elif mode=="loo":
                ## 1.0 no noise, 0.0 is noise
                mask_p_binary = tf.cast(tf.where(tf.math.logical_and(mask_p_full/d>(th-ths_step),mask_p_full/d<=th),1.0,0.0),tf.float64).numpy()
                mask_p = tf.cast(tf.where(tf.math.logical_and(mask_p_full/d>(th-ths_step),mask_p_full/d<=th),1.0,0.5),tf.float64).numpy()
            else:
                if manual_th != "full":
                    mask_p_binary = tf.cast(tf.where((mask_p_full/d)>th,1.0,0.0),tf.float64).numpy()
                    mask_p = tf.cast(tf.where((mask_p_full/d)>th,1.0,0.5),tf.float64).numpy()
                else:
                    mask_p_binary = np.ones_like(mask_p_full)
                    mask_p = mask_p_full/d
            mask_size = np.sum(mask_p_binary,dtype=np.float64)
            mask_organelle_intersection = np.sum(mask_p_binary*target_seg_image,dtype=np.float64)/mask_size
            mask_size = mask_size/np.prod(mask_p_binary.shape)
            
            ## Create noisy input and predict unet
            mask_patchs_p_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p, gv.patch_size, xy_step, z_step)
            mask_noise_patchs = (normal_noise*(1-mask_patchs_p_term))
            input_patchs_p = (mask_patchs_p_term*input_patchs)+mask_noise_patchs
            unet_noise_patchs_p = predict(model.unet,input_patchs_p.numpy(),batch_size,gv.patch_size)#model.unet.predict(tf.convert_to_tensor(input_patchs_p),batch_size=batch_size)
    
            ## Back to image 
            input_p,unet_noise_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p],weights,input_image.shape,gv.patch_size,xy_step,z_step)
            
            ## Save images
            if save_image:
                print("saving mask images...")
                base_save = "{}/{}/{}/".format(dir_path,image_index,th)
                ImageUtils.imsave((mask_p).astype(np.float16),"{}/mask_{}.tiff".format(base_save,image_index))
                ImageUtils.imsave((input_p/d).astype(np.float16),"{}/noisy_input_{}.tiff".format(base_save,image_index))
                ImageUtils.imsave((unet_noise_p/d).astype(np.float16),"{}/noisy_unet_prediction_{}.tiff".format(base_save,image_index))  
            # pcc = pearson_corr((unet_p*mem_seg_image/d)[:,:,:], (unet_noise_p*mem_seg_image/d)[:,:,:])
            pcc = pearson_corr((unet_p/d)[:,:,:], (unet_noise_p/d)[:,:,:],target_seg_image_dilated)
            pcc_gt = pearson_corr((unet_p/d)[:,:,:], (target_image)[:,:,:],target_seg_image_dilated)
            pcc_gt_noise = pearson_corr((unet_noise_p/d)[:,:,:], (target_image)[:,:,:],target_seg_image_dilated)
            pccs.append(pcc)
            mask_sizes.append(mask_size)
            c = 1/(mask_organelle_intersection/mask_size)
            context.append(c)
            print("pearson corr for image:{} is :{}, mask ratio:{} context:{}".format(image_index,pcc,mask_size,c))
            print("pearson corr for gt image:{} is :{}, mask ratio:{} context:{}".format(image_index,pcc_gt,mask_size,c))
            print("pearson corr for noise gt image:{} is :{}, mask ratio:{} context:{}".format(image_index,pcc_gt_noise,mask_size,c))
                       
            del unet_noise_patchs_p
            del input_p
            del unet_noise_p
            del mask_noise_patchs
            del mask_patchs_p_term
            
        if save_image:
            print("saving global images...")
            image_save = "{}/{}/".format(dir_path,image_index)
            ImageUtils.imsave((input_image).astype(np.float16),"{}/input_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave((target_image).astype(np.float16),"{}/target_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(nuc_image.astype(np.float16),"{}/nuc_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(mem_image.astype(np.float16),"{}/mem_{}.tiff".format(image_save,image_index))  
            ImageUtils.imsave((unet_p/d).astype(np.float16),"{}/unet_prediction_{}.tiff".format(image_save,image_index))   
            ImageUtils.imsave((target_seg_image).astype(np.float16),"{}/seg_target_{}.tiff".format(image_save,image_index))
        pcc_results.add_row(pccs)
        mask_results.add_row(mask_sizes)
        context_results.add_row(context)
        if save_histo:
            h, _ = np.histogram((mask_p).reshape(1,-1), bins=num_bins, density=True) #ignore 0.0 for single cells
            sum_histos += h
            histos.add_row(h)
            histos_axs[count // histos_l, count % histos_l].plot(bins,h)
            histos_axs[count // histos_l, count % histos_l].set_title(image_index)
            histos.create()
            histos_fig.savefig("{}/histograms.png".format(dir_path))
        count+=1   
    pcc_results.create()
    mask_results.create()
    context_results.create()
    if save_histo:
        avg_histo = sum_histos/len(images)
        histos_axs[(count) // histos_l, (count) % histos_l].plot(bins,avg_histo)
        histos_axs[(count) // histos_l, (count) % histos_l].set_title("average")    
        histos.add_row(avg_histo)
        histos.create()
        histos_fig.savefig("{}/histograms.png".format(dir_path))

def analyze_correlations_constant(organelles,mask_th,save_images=True,absoulte_organelle_precent_pixels=0.05,compound=None,is_vehicle=False,model_path=gv.model_path):
    """This method is used to evaluate the importance of pixels of a certain organelle W.R.T random pixels.

    Args:
        organelles (list): list of organelles to check
        mask_th (float): the th to use
        save_images (bool, int,optional): if True save all images, if False save nothing, if int, save the first images according to the int. Defaults to True. Defaults to True.
        absoulte_organelle_precent_pixels (float, optional):  noise percent of the mask, for example if absoulte_organelle_precent_pixels = 0.05, then the noise will be 5% from the amount of pixels in the mask, but the pixels will be chosen from the organelle we want to check. Defaults to 0.05.
        compound (str, optional): if not None will use perturbation DS. Defaults to None.
        is_vehicle (bool, optional): DMSO or drug in the perturbation DS. Defaults to False.
        model_path (str, optional): _description_. Defaults to gv.model_path.
    """
    ## Create main dir and organelle subdir
    pred_path = "predictions_correlations_constant"
        
    base_dir_path = "{}/{}".format(model_path,pred_path)
    create_dir_if_not_exist(base_dir_path)
    repeat = 1 # number of repates of sampling random pixels.
    for organelle in organelles:
        print(organelle)
        if (compound is not None):
            if is_vehicle:
                ov = "Vehicle"
            else:
                ov = compound
            dir_path = "{}/{}_{}".format(base_dir_path,organelle,compound,ov)
        else:
            dir_path = "{}/{}".format(base_dir_path,organelle.replace(' ','-'))
            
        create_dir_if_not_exist(dir_path)
    
        if is_vehicle:
            drug = "Vehicle"
        else:
            drug = compound
            
        if compound is not None:
            ds_path = "/sise/home/lionb/single_cell_training_from_segmentation_pertrub/{}_{}/image_list_test_{}.csv".format(organelle,compound,drug)
        else:
            ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(organelle.replace(' ','-'))
    
        corr_results = DatasetMetadataSCV("{}/corr_results_{}_{}_organelle_precent_pixels_{}_in_image.csv".format(dir_path,organelle, drug, absoulte_organelle_precent_pixels))
        corr_results.create_header(["ratio_of_organelle_in_image","ratio_of_mask_in_image","ratio_of_mask_organelle_intersection_in_mask","ratio_of_mask_organelle_intersection_in_organelle","ratio_of_mask_organelle_intersection_in_image","ratio_of_noised_pixels_wo_mask_in_image","ratio_of_noised_pixels_with_mask_in_image","ratio_of_noised_random_flip_in_image","ratio_of_noised_random_pixels_in_image","pcc","pcc_wo_organelle_pixels","pcc_wo_organelle_pixels_wo_mask","pcc_random_flip","pcc_random_pixels"])    
        
        try:
            dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
        except:
            print ("{} not exist, continue...".format(ds_path))
            continue
        
        images = range(min([dataset.df.get_shape()[0],10])) #
        item = 0
        for image_index in images:

            tf.keras.backend.clear_session()
            _ = gc.collect()
            
            print("image index: {}/{}".format(image_index+1,min([dataset.df.get_shape()[0],10])))
            
            ## Preprocess images
            px_start=center_xy[0]-margin[0]-xy_step
            py_start=center_xy[1]-margin[1]-xy_step
            pz_start = 0
            px_end=center_xy[0]+margin[0]+xy_step
            py_end=center_xy[1]+margin[1]+xy_step
            
            
            slice_by = [None,(px_start,px_end),(py_start,py_end)]
            print("preprocess..")
            input_image,target_image,target_seg_image,nuc_image,mem_image = preprocess_image(dataset,image_index,[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane"],normalize=[True,True,False,True,True],slice_by=slice_by)
                
            target_seg_image = target_seg_image/255.
            pz_end = input_image.shape[0]
            
            weights = None
            
            ## Collect patchs
            input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,input_image, gv.patch_size, xy_step, z_step)
            
            ## Batch predict
            ## Predicte unet and mask
            print("batch predict...")
            unet_patchs_p = predict(model.unet,input_patchs,batch_size,gv.patch_size)
            mask_patchs_p = predict(model,input_patchs,batch_size,gv.patch_size)
            
            ## Create noise vector
            normal_noise = tf.random.normal(tf.shape(mask_patchs_p),stddev=noise_scale,dtype=tf.float32)
            # normal_noise = tf.random.uniform(tf.shape(mask_patchs_p),maxval=noise_scale,dtype=tf.float64)
            
            ## Back to image 
            weights = get_weights(input_patchs[0].shape)
            unet_p,mask_p_full,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,mask_patchs_p,np.ones_like(mask_patchs_p)],weights,input_image.shape,gv.patch_size,xy_step,z_step)
            mask_p_full = mask_p_full/d
            
            base_save = "{}/{}/".format(dir_path,image_index)
            if image_index<save_images:
                ## Create image dir
                create_dir_if_not_exist(base_save)
                print("saving global images...")
                ImageUtils.imsave(input_image.astype(np.float16),"{}/input_{}.tiff".format(base_save,image_index))
                ImageUtils.imsave(target_image.astype(np.float16),"{}/target_{}.tiff".format(base_save,image_index))
                ImageUtils.imsave(nuc_image.astype(np.float16),"{}/nuc_{}.tiff".format(base_save,image_index)) 
                ImageUtils.imsave(mem_image.astype(np.float16),"{}/mem_{}.tiff".format(base_save,image_index))  
                ImageUtils.imsave((unet_p/d).astype(np.float16),"{}/unet_prediction_{}.tiff".format(base_save,image_index))                  
            for r in range(repeat):  
                ## generate new noise vector
                normal_noise = tf.random.normal(tf.shape(mask_patchs_p),stddev=noise_scale,dtype=tf.float32)              
                print("mask predict ...")
                mask_p_binary = tf.where(mask_p_full>mask_th,1.0,0.0)
                mask_p = tf.where((mask_p_binary)>mask_th,1.0,0.5)
                mask_organelle_itersection = (mask_p_binary * target_seg_image)
                ratio_of_organelle_in_image = np.sum(target_seg_image,dtype=np.float64)/np.prod(target_seg_image.shape)
                
                ratio_of_mask_organelle_intersection_in_image = np.sum(mask_organelle_itersection,dtype=np.float64)/np.prod(mask_organelle_itersection.shape)
                ratio_of_mask_in_image = np.sum(mask_p_binary,dtype=np.float64)/np.prod(mask_p_binary.shape)
                ratio_of_mask_organelle_intersection_in_mask = np.sum(mask_organelle_itersection,dtype=np.float64)/np.sum(mask_p_binary,dtype=np.float64)
                
                ratio_of_noise_pixels_in_image = absoulte_organelle_precent_pixels
                organelle_precent_pixels = absoulte_organelle_precent_pixels/ratio_of_mask_organelle_intersection_in_image
                ax = r #np.random.randint(0,3)
                target_seg_image_flip = np.flip(target_seg_image,axis=ax)  
                ratio_of_organelle_in_image_flip = np.sum((target_seg_image_flip*(1-target_seg_image)),dtype=np.float64)/np.prod(target_seg_image.shape)
                
                if organelle_precent_pixels > 1.0 or ratio_of_noise_pixels_in_image/ratio_of_organelle_in_image > 1.0 or ratio_of_noise_pixels_in_image/ratio_of_organelle_in_image_flip > 1.0:
                    item +=1
                    continue
                organelle_pixels_target_seg_image = np.random.choice([0.0,1.],target_seg_image.shape,p=[1-(organelle_precent_pixels),organelle_precent_pixels])
                mask_p_binary_wo_organelle_pixels = 1 - organelle_pixels_target_seg_image*mask_organelle_itersection #+ (1-target_seg_image) + (1-mask_p_binary),0.0)
                mask_p_wo_organelle_pixels = tf.where(mask_p_binary_wo_organelle_pixels>0.0,1.0,0.5)
                
                organelle_pixels_target_seg_image = np.random.choice([0.0,1.],target_seg_image.shape,p=[1-(ratio_of_noise_pixels_in_image/ratio_of_organelle_in_image),ratio_of_noise_pixels_in_image/ratio_of_organelle_in_image])
                wo_organelle_pixels_binary = tf.math.minimum((target_seg_image * (1 - organelle_pixels_target_seg_image))+((1-target_seg_image)),1.0)
                wo_organelle_pixels = tf.where(wo_organelle_pixels_binary>0.0,1.0,0.5)  
                
                # random_pixels_binary = np.random.choice([1.0,0.0],target_seg_image.shape,p=[1-(absoulte_organelle_precent_pixels),absoulte_organelle_precent_pixels])
                # random_pixels = tf.where(random_pixels_binary>0.0,1.0,0.0)  
                
                organelle_pixels_target_seg_image = np.random.choice([0.0,1.],target_seg_image.shape,p=[1-(ratio_of_noise_pixels_in_image/ratio_of_organelle_in_image_flip),ratio_of_noise_pixels_in_image/ratio_of_organelle_in_image_flip])
                random_flip_binary = tf.math.minimum(((1-target_seg_image)*target_seg_image_flip * (1 - organelle_pixels_target_seg_image))+((1-target_seg_image_flip))+(target_seg_image*target_seg_image_flip),1.0)
                random_flip = tf.where(random_flip_binary>0.0,1.0,0.5)                                              
                del organelle_pixels_target_seg_image
                del target_seg_image_flip
                
                random_pixels_binary = np.random.choice([0.0,1.],target_seg_image.shape,p=[ratio_of_noise_pixels_in_image,1-ratio_of_noise_pixels_in_image])
                random_pixels = tf.where(random_pixels_binary>0.0,1.0,0.5)
                
                ## Create noisy input and predict unet
                mask_patchs_p_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p, gv.patch_size, xy_step, z_step)
                mask_patchs_p_wo_organelle_pixels_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p_wo_organelle_pixels, gv.patch_size, xy_step, z_step)
                wo_organelle_pixels_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,wo_organelle_pixels, gv.patch_size, xy_step, z_step)
                random_flip_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,random_flip, gv.patch_size, xy_step, z_step)
                random_pixels_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,random_pixels, gv.patch_size, xy_step, z_step)
                
                mask_noise_patchs = (normal_noise*(1-mask_patchs_p_term))
                mask_noise_wo_organelle_pixels_patchs = (normal_noise*(1-mask_patchs_p_wo_organelle_pixels_term))
                wo_organelle_pixels_patchs = (normal_noise*(1-wo_organelle_pixels_term))
                random_flip_patchs = (normal_noise*(1-random_flip_term))
                random_pixels_patchs = (normal_noise*(1-random_pixels_term))
                
                input_patchs_p = (mask_patchs_p_term*input_patchs)+mask_noise_patchs
                input_patchs_wo_organelle_pixels_p = (mask_patchs_p_wo_organelle_pixels_term*input_patchs)+mask_noise_wo_organelle_pixels_patchs
                input_patchs_wo_organelle_pixels_wo_mask_p = (wo_organelle_pixels_term*input_patchs)+wo_organelle_pixels_patchs
                input_patchs_random_flip_p = (random_flip_term*input_patchs)+random_flip_patchs
                input_patchs_random_pixels_p = (random_pixels_term*input_patchs)+random_pixels_patchs
                
                unet_noise_patchs_p = predict(model.unet,input_patchs_p.numpy(),batch_size,gv.patch_size)
                unet_noise_patchs_wo_organelle_pixels_p = predict(model.unet,input_patchs_wo_organelle_pixels_p.numpy(),batch_size,gv.patch_size)
                unet_noise_patchs_wo_organelle_pixels_wo_mask_p = predict(model.unet,input_patchs_wo_organelle_pixels_wo_mask_p.numpy(),batch_size,gv.patch_size)
                unet_noise_patchs_random_flip_p = predict(model.unet,input_patchs_random_flip_p.numpy(),batch_size,gv.patch_size)
                unet_noise_patchs_random_pixels_p = predict(model.unet,input_patchs_random_pixels_p.numpy(),batch_size,gv.patch_size)
                
                ## Back to image 
                input_p,unet_noise_p,unet_noise_wo_organelle_pixels_p,unet_noise_wo_organelle_pixels_wo_mask_p,unet_noise_random_flip_p,unet_noise_random_pixels_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p,unet_noise_patchs_wo_organelle_pixels_p,unet_noise_patchs_wo_organelle_pixels_wo_mask_p,unet_noise_patchs_random_flip_p,unet_noise_patchs_random_pixels_p],weights,input_image.shape,gv.patch_size,xy_step,z_step)
                
                ## Save images
                if image_index<save_images:
                    print("saving mask images...")
                    ImageUtils.imsave(mask_p_binary.numpy().astype(np.float16),"{}/mask_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave(mask_p_binary_wo_organelle_pixels.numpy().astype(np.float16),"{}/mask_wo_organelle_pixels_{}.tiff".format(base_save,image_index))            
                    ImageUtils.imsave(target_seg_image.astype(np.float16),"{}/seg_source_organelle_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((input_p/d).astype(np.float16),"{}/noisy_input_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((unet_noise_p/d).astype(np.float16),"{}/noisy_unet_prediction_{}.tiff".format(base_save,image_index)) 
                    ImageUtils.imsave((unet_noise_wo_organelle_pixels_p/d).astype(np.float16),"{}/noisy_unet_prediction_wo_organelle_pixels_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((unet_noise_wo_organelle_pixels_wo_mask_p/d).astype(np.float16),"{}/noisy_unet_prediction_wo_organelle_pixels_wo_mask_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((unet_noise_random_flip_p/d).astype(np.float16),"{}/noisy_unet_prediction_random_flip_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((unet_noise_random_pixels_p/d).astype(np.float16),"{}/noisy_unet_prediction_random_pixels_{}.tiff".format(base_save,image_index))

                pcc = pearson_corr((unet_p/d)[:,:,:], (unet_noise_p/d)[:,:,:])
                pcc_wo_organelle_pixels = pearson_corr((unet_p/d)[:,:,:], (unet_noise_wo_organelle_pixels_p/d)[:,:,:])
                pcc_wo_organelle_pixels_wo_mask = pearson_corr((unet_p/d)[:,:,:], (unet_noise_wo_organelle_pixels_wo_mask_p/d)[:,:,:])
                pcc_random_flip = pearson_corr((unet_p/d)[:,:,:], (unet_noise_random_flip_p/d)[:,:,:])
                pcc_random_pixels = pearson_corr((unet_p/d)[:,:,:], (unet_noise_random_pixels_p/d)[:,:,:])
                
                corr_results.set_item(item,"ratio_of_organelle_in_image",ratio_of_organelle_in_image)
                corr_results.set_item(item,"ratio_of_mask_in_image",ratio_of_mask_in_image)
                corr_results.set_item(item,"ratio_of_mask_organelle_intersection_in_image",ratio_of_mask_organelle_intersection_in_image)
                corr_results.set_item(item,"ratio_of_mask_organelle_intersection_in_mask",ratio_of_mask_organelle_intersection_in_mask)
                corr_results.set_item(item,"ratio_of_mask_organelle_intersection_in_organelle",np.sum(mask_organelle_itersection,dtype=np.float64)/np.sum(target_seg_image,dtype=np.float64))
                # corr_results.set_item(item,"ratio_of_noised_pixels_in_mask",1 - np.sum(mask_p_binary_wo_organelle_pixels.numpy(),dtype=np.float64)/np.sum(mask_p_binary,dtype=np.float64))
                corr_results.set_item(item,"ratio_of_noised_pixels_wo_mask_in_image",(1-np.sum(wo_organelle_pixels_binary.numpy(),dtype=np.float64)/np.prod(wo_organelle_pixels_binary.shape)))
                corr_results.set_item(item,"ratio_of_noised_pixels_with_mask_in_image",(1-np.sum(mask_p_binary_wo_organelle_pixels.numpy(),dtype=np.float64)/np.prod(wo_organelle_pixels_binary.shape)))
                corr_results.set_item(item,"ratio_of_noised_random_flip_in_image",(1-np.sum(random_flip,dtype=np.float64)/np.prod(wo_organelle_pixels_binary.shape)))
                corr_results.set_item(item,"ratio_of_noised_random_pixels_in_image",(1-np.sum(random_pixels,dtype=np.float64)/np.prod(wo_organelle_pixels_binary.shape)))
                
        
                corr_results.set_item(item,"pcc",pcc)
                corr_results.set_item(item,"pcc_wo_organelle_pixels",pcc_wo_organelle_pixels)
                corr_results.set_item(item,"pcc_wo_organelle_pixels_wo_mask",pcc_wo_organelle_pixels_wo_mask)
                corr_results.set_item(item,"pcc_random_flip",pcc_random_flip)
                corr_results.set_item(item,"pcc_random_pixels",pcc_random_pixels) 
                
                importance_in_organelle = np.sum(mask_p_full * target_seg_image,dtype=np.float64)/np.sum(target_seg_image,dtype=np.float64)
                corr_results.set_item(item,"importance_in_organelle",importance_in_organelle)
                
                print("pearson corr for image to mask:{} reapet:{} is :{} wo organelle pixels in mask:{}, wo organelle pixels:{}, random flip:{}, random pixels:{}, importance_in_organelle:{}".format(image_index,r,pcc,pcc_wo_organelle_pixels,pcc_wo_organelle_pixels_wo_mask,pcc_random_flip,pcc_random_pixels,importance_in_organelle)) 
                     
                item +=1
                del input_patchs_p
                del input_patchs_wo_organelle_pixels_p  
                del input_patchs_wo_organelle_pixels_wo_mask_p 
                del unet_noise_patchs_wo_organelle_pixels_wo_mask_p            
                del unet_noise_patchs_p
                del unet_noise_wo_organelle_pixels_p
                del input_p
                del unet_noise_p
                del mask_noise_patchs
                del mask_noise_wo_organelle_pixels_patchs
                del wo_organelle_pixels_patchs
                del mask_p_wo_organelle_pixels
                del wo_organelle_pixels
                del random_flip
                del random_flip_patchs
                del input_patchs_random_flip_p
                del unet_noise_patchs_random_flip_p
                del unet_noise_random_flip_p
                del random_pixels
                del random_pixels_patchs
                del input_patchs_random_pixels_p
                del unet_noise_patchs_random_pixels_p
                del unet_noise_random_pixels_p
                del normal_noise
                
            
            del unet_patchs_p
            del mask_patchs_p
        corr_results.create()
        
def analyze_predictions(organelle,save_images=True,compound=None,is_vehicle=False,model_path=gv.model_path,model=None):
    """Calculate PCC between orininal and noisy input predictions for images of an organelle.

    Args:
        organelle (str): name of organelle
        save_images (bool, int,optional): if True save all images, if False save nothing, if int, save the first images according to the int. Defaults to True. Defaults to True.
        compound (str, optional): if not None use perturbation DS. Defaults to None.
        is_vehicle (bool, optional): if compound is not None, and is_vehicle is True use DMSO from perturbation DS, if is_vehicle is False use drug data. Defaults to False.
        model_path (str, optional): use to load model if not passed and dir to save results. Defaults to gv.model_path.
        model (model, optional): model if exists, if None method will load model from model_path. Defaults to None.
    """
    if model is None:
        print("Loading model:",param["model"])
        model = keras.models.load_model(param["model"])
        
    ## Create main dir and organelle subdir
    pred_path = "predictions"
        
    base_dir_path = "{}/{}".format(model_path,pred_path)
    create_dir_if_not_exist(base_dir_path)
    organelle = organelle
    print(organelle,is_vehicle)
    if (compound is not None):
        if is_vehicle:
            ov = "Vehicle"
        else:
            ov = compound
        dir_path = "{}/{}_{}".format(base_dir_path,organelle,compound,ov)
    else:
        dir_path = "{}/{}".format(base_dir_path,organelle)
        
    create_dir_if_not_exist(dir_path)

    if is_vehicle:
        drug = "Vehicle"
    else:
        drug = compound
        
    if compound is not None:
        ds_path = "/sise/home/lionb/single_cell_training_from_segmentation_pertrub/{}_{}/image_list_test_{}.csv".format(organelle.replace('-',' '),compound,drug)
    else:
        if is_vehicle:
            return
        organelle = organelle.replace(' ','-')
        ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(organelle)

    pred_results = DatasetMetadataSCV("{}/predictions_results_{}_{}_in_image.csv".format(dir_path,organelle, drug))
    pred_results.create_header(["pcc"])    
    
    try:
        dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
    except:
        print ("{} not exist, continue...".format(ds_path))
        return
    
    images = range(min([dataset.df.get_shape()[0],10]))
    item = 0
    for image_index in images:

        tf.keras.backend.clear_session()
        _ = gc.collect()
        
        print("image index: {}/{}".format(image_index+1,min([dataset.df.get_shape()[0],10])))
        
        ## Preprocess images
        px_start=center_xy[0]-margin[0]-xy_step
        py_start=center_xy[1]-margin[1]-xy_step
        pz_start = 0
        px_end=center_xy[0]+margin[0]+xy_step
        py_end=center_xy[1]+margin[1]+xy_step
        
        
        slice_by = [None,(px_start,px_end),(py_start,py_end)]
        print("preprocess..")
        input_image,target_image,target_seg_image,nuc_image,mem_image = preprocess_image(dataset,image_index,[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane"],normalize=[True,True,False,True,True],slice_by=slice_by)
            
        target_seg_image = target_seg_image/255.
        pz_end = input_image.shape[0]
        
        weights = None
        
        ## Collect patchs
        input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,input_image, gv.patch_size, xy_step, z_step)
        
        ## Batch predict
        ## Predicte unet and mask
        print("batch predict...")
        unet_patchs_p = predict(model,input_patchs,batch_size,gv.patch_size)
        
        ## Back to image 
        weights = get_weights(input_patchs[0].shape)
        unet_p,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,np.ones_like(unet_patchs_p)],weights,input_image.shape,gv.patch_size,xy_step,z_step)
        
        base_save = "{}/{}/".format(dir_path,image_index)
        if image_index<save_images:
            ## Create image dir
            create_dir_if_not_exist(base_save)
            print("saving global images...")
            ImageUtils.imsave(input_image.astype(np.float16),"{}/input_{}.tiff".format(base_save,image_index))
            ImageUtils.imsave(target_image.astype(np.float16),"{}/target_{}.tiff".format(base_save,image_index))
            ImageUtils.imsave(nuc_image.astype(np.float16),"{}/nuc_{}.tiff".format(base_save,image_index)) 
            ImageUtils.imsave(mem_image.astype(np.float16),"{}/mem_{}.tiff".format(base_save,image_index))  
            ImageUtils.imsave((unet_p/d).astype(np.float16),"{}/unet_prediction_{}.tiff".format(base_save,image_index))                 
        
        pcc = pearson_corr((unet_p/d)[:,:,:], (target_image)[:,:,:])
        pred_results.set_item(item,"pcc",pcc)      
        print("pearson corr for image:{} is :{}".format(image_index,pcc))             
        item +=1

        del input_patchs
        del unet_patchs_p
        
    pred_results.create()

# analyze_correlations(save_images=3, mask_th=0.7, organelle_precent_pixels=0.5, organelles=["Plasma-membrane","Nuclear-envelope","Endoplasmic-reticulum","Nucleolus-(Dense-Fibrillar-Component)","Nucleolus-(Granular-Component)","Microtubules","Tight-junctions","Mitochondria","Actomyosin-bundles","Actin-filaments","Desmosomes","Golgi"])
# analyze_correlations_constant(save_images=3, mask_th=0.6, absoulte_organelle_precent_pixels=0.001, organelles=["Peroxisomes","Endosomes"])#["Nuclear-envelope","Endoplasmic-reticulum","Golgi","Plasma-membrane","Nucleolus-(Dense-Fibrillar-Component)","Nucleolus-(Granular-Component)","Microtubules","Tight-junctions","Mitochondria","Actomyosin-bundles","Actin-filaments","Desmosomes","Lysosome","Adherens-junctions","Gap-junctions","Matrix-adhesions","Peroxisomes","Endosomes"])
# analyze_correlations_constant(save_images=3, mask_th=0.65, absoulte_organelle_precent_pixels=0.001, organelles=["Mitochondria","Actomyosin-bundles","Actin-filaments","Desmosomes","Lysosome","Adherens-junctions","Gap-junctions","Matrix-adhesions","Peroxisomes","Endosomes"])
# analyze_correlations_constant(save_images=3, mask_th=0.8, absoulte_organelle_precent_pixels=0.03, organelles=["Microtubules"])

# analyze_th("regular",save_histo=False,save_image=1,weighted_pcc=weighted_pcc)
# analyze_th(mode="agg",mask_image=None,manual_th="full",save_image=1,save_histo=False,weighted_pcc=weighted_pcc)
# for mth in [0.85]:
    # analyze_th("regular",mask_image=None,manual_th=mth,save_image=True,save_histo=False)
# analyze_th("mask","{}/X_gradcam_layer_downsample_4_full.tiff".format(gv.model_path)) #gbp_full.tiff #saliency_full.tiff #X_gradcam_layer_downsample_4_full.tiff
# analyze_th("mask","{}/predictions_agg/1/MASK_4.tif".format(gv.model_path))
# analyze_th("mask","{}/predictions_masked/4/MASK_2.tif".format(gv.model_path))
# analyze_th("agg")
# analyze_th("loo")

# analyze_correlations(save_images=3,mask_th=0.85,organelles=["Endoplasmic-reticulum","Plasma-membrane","Actin-filaments"])
# analyze_correlations(save_images=3,mask_th=0.85,organelles=["Golgi","Microtubules","Nucleolus-(Dense-Fibrillar-Component)"])
# analyze_correlations(save_images=3,mask_th=0.85,organelles=["Mitochondria","Tight-junctions","Nucleolus-(Granular-Component)"])
# analyze_correlations(save_images=3,mask_th=0.85,organelles=["Actomyosin-bundles","Nuclear-envelope","Desmosomes",Nucleolus-(Dense-Fibrillar-Component)"])

##perturbations
# params = [\
#     {"model":"./mg_model_dna_10_06_22_5_0_dnab3","th":0.5},\
#     # {"model":"./mg_model_dna_10_06_22_5_0_mlw_0.1","th":0.5},\
#     # {"model":"./mg_model_er_10_06_22_5_0_mlw_0.17","th":0.4},\
#         # {"model":"./mg_model_microtubules_10_06_22_5_0_new","th":0.65},\
#             # {"model":"./mg_model_actin_10_06_22_5_0_new","th":0.8},\
#                 {"model":"./mg_model_mito_10_06_22_5_0_new","th":0.75},\
#                     # {"model":"./mg_model_membrane_10_06_22_5_0_new","th":0.8},\
#                         {"model":"./mg_model_ne_10_06_22_5_0_new","th":0.6},\
#                             # {"model":"./mg_model_ngc_10_06_22_5_0_new","th":0.5},\
#                                 {"model":"./mg_model_golgi_10_06_22_5_0_new","th":0.65}
#         ]
# params = [{"model":"./mg_model_ne_10_06_22_5_0_new","th":0.6},\
#             {"model":"./mg_model_ngc_10_06_22_5_0_new","th":0.5},\
#                 {"model":"./mg_model_golgi_10_06_22_5_0_new","th":0.65}]

#predictions
# params = [{"model":"./unet_model_22_05_22_mito_128","organelle":"Mitochondria"},\
#             {"model":"./unet_model_22_05_22_membrane_128","organelle":"Plasma-membrane"},\
#             {"model":"./unet_model_22_05_22_ne_128","organelle":"Nuclear-envelope"},\
#             {"model":"./unet_model_22_05_22_ngc_128","organelle":"Nucleolus-(Granular-Component)"},\
#             {"model":"./unet_model_22_05_22_er_128","organelle":"Endoplasmic-reticulum"},\
#             {"model":"./unet_model_22_05_22_actin_128","organelle":"Actin filaments"},\
#             {"model":"./unet_model_22_05_22_bundles_128","organelle":"Actomyosin bundles"},\
#             {"model":"./unet_model_22_05_22_golgi_128","organelle":"Golgi"},\
#             {"model":"./unet_model_22_05_22_microtubules_128","organelle":"Microtubules"},\
#             {"model":"./unet_model_22_05_22_tj_128","organelle":"Tight junctions"}\
#             ]


# organelles = ["Lysosome","Adherens-junctions","Gap-junctions","Matrix-adhesions","Golgi","Microtubules","Endoplasmic-reticulum","Plasma-membrane","Actin-filaments","Peroxisomes","Endosomes","Nucleolus-(Granular-Component)","Mitochondria","Tight-junctions","Actomyosin-bundles","Nuclear-envelope","Desmosomes","Nucleolus-(Dense-Fibrillar-Component)"]
# for organelle in organelles:
#     params = [{"model":"./unet_model_22_05_22_dna_128b","organelle":"Endosomes"}]
#     gv.target = "channel_dna"
#     # Load model
#     compounds = ["Rapamycin","Paclitaxol","Staurosporine","Brefeldin"]#[None,"s-Nitro-Blebbistatin","Rapamycin","Paclitaxol","Staurosporine","Brefeldin"]#,"s-Nitro-Blebbistatin","Rapamycin","Paclitaxol","Staurosporine","Brefeldin"]
#     for compound in compounds:
#         print(compound)
#         for param in params: 
#             tf.keras.backend.clear_session()
#             print("Loading model:",param["model"])
#             model = keras.models.load_model(param["model"])
#             for is_vehicle in [False,True]:
#                 # analyze_correlations_constant(save_images=3,absoulte_organelle_precent_pixels=0.001, mask_th=param["th"], compound=compound, is_vehicle=is_vehicle, organelles=["Actin filaments","Endoplasmic reticulum","Lysosome","Microtubules","Tight junctions","Golgi","Actomyosin bundles"],model_path=param["model"])
#                 # analyze_correlations_constant(save_images=3,absoulte_organelle_precent_pixels=0.000, mask_th=param["th"], compound=compound, is_vehicle=is_vehicle, organelles=["Lysosome","Adherens-junctions","Gap-junctions","Matrix-adhesions","Golgi","Microtubules","Endoplasmic-reticulum","Plasma-membrane","Actin-filaments","Peroxisomes","Endosomes","Nucleolus-(Granular-Component)","Mitochondria","Tight-junctions","Actomyosin-bundles","Nuclear-envelope","Desmosomes","Nucleolus-(Dense-Fibrillar-Component)"],model_path=param["model"])
#                 # analyze_correlations_constant(save_images=4,absoulte_organelle_precent_pixels=0.000, mask_th=param["th"], compound=compound, is_vehicle=is_vehicle, organelles=["Endosomes"],model_path=param["model"])
#                 analyze_predictions(organelle,save_images=4,compound=compound,is_vehicle=is_vehicle,model_path=param["model"],model = model)
                    
            
            
            
        
        
        
        

                

