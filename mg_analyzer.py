from copy import deepcopy
import gc
from numpy import dtype
from scipy import signal
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
from metrics import *
from cell_imaging_utils.image.image_utils import ImageUtils
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
import global_vars as gv
from utils import *
import os

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

def predict(model,data,batch_size):    
    batch_data = data.reshape((-1,batch_size,*gv.patch_size))
    output = np.zeros_like(batch_data)
    for i in range(batch_data.shape[0]):
        batch_pred = model.predict_on_batch(batch_data[i])
        output[i] = batch_pred
    output = output.reshape(data.shape)
    K.clear_session()
    _ = gc.collect()
    return output

def _get_weights(shape):
    shape_in = shape
    shape = shape[1:]
    weights = 1
    for idx_d in range(len(shape)):
        slicey = [np.newaxis] * len(shape)
        slicey[idx_d] = slice(None)
        size = shape[idx_d]
        values = signal.triang(size)
        weights = weights * values[tuple(slicey)] #scipy.ndimage.gaussian_filter(values, sigma=1)[tuple(slicey)]
        
        #weights = weights * np.ones(size)[tuple(slicey)]
    return np.broadcast_to(weights, shape_in).astype(np.float32)

# dataset = DataGen(gv.test_ds_path, gv.input, gv.target, batch_size=1, num_batches=1, patch_size=gv.patch_size, min_precentage=0, max_precentage=1, augment=False)
dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=0.9, augment=False)

## Choose images
images = range(30)#range(dataset.df.get_shape()[0])

## Noise
noise_scale = 5.0

## Batch size
batch_size=9

## Image center
center_xy = [312,465]
margin=[192,256]

## Patch steps
xy_step = 64
z_step = 16

## Load model
print("Loading model:",gv.mg_model_path)
model = keras.models.load_model(gv.mg_model_path)

def collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,image):
    # Init indexes
    pz = pz_start
    px = px_start
    py = py_start
    
    ## Collect patchs
    print("collect patchs...")
    patchs = []
    while pz<=pz_end-gv.patch_size[0]:
        while px<=px_end-gv.patch_size[1]:
            while py<=py_end-gv.patch_size[2]: 
                
                ## Slice patch from input       
                px_start_patch = px-px_start
                py_start_patch = py-py_start    
                s = [(pz,pz+gv.patch_size[0]),(px_start_patch,px_start_patch+gv.patch_size[1]),(py_start_patch,py_start_patch+gv.patch_size[2])]
                patch = slice_image(image,s)
                # seg_patch = slice_image(target_seg_image,s)
                
                patchs.append(patch)
                
                py+=xy_step
            py=py_start  
            px+=xy_step
        px=px_start
        pz+=z_step
    return np.array(patchs)

def assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,patchs,weights,assembled_image_shape):
    print("assemble images from patchs...")
    patchs = np.array(patchs)
    assembled_images = np.zeros((patchs.shape[0],*assembled_image_shape))+0.00001
    pz = pz_start
    px = px_start
    py = py_start 
    i = 0
    while pz<=pz_end - gv.patch_size[0]:
        while px<=px_end-gv.patch_size[1]:
            while py<=py_end-gv.patch_size[2]: 
                px_start_patch = px-px_start
                py_start_patch = py-py_start 
                
                ## Update with patchs
                patch_slice = (slice(pz,pz+gv.patch_size[0]), slice(px_start_patch,px_start_patch+gv.patch_size[1]),slice(py_start_patch,py_start_patch+gv.patch_size[2]))
                
                for j in range(patchs.shape[0]):
                    assembled_images[j][patch_slice] += patchs[j][i]*weights 
                
                py+=xy_step
                i+=1
            py=py_start  
            px+=xy_step
        px=px_start
        pz+=z_step
    return assembled_images

def analyze_th(mode,mask_image=None):
    ## Create thresholds
    ths_start = 0.5
    ths_step = 0.1
    ths_stop = 1.1
    ths = np.arange(ths_start,ths_stop,ths_step)
    ## Create main dir
    if mode=="agg":
        pred_path = "predictions_agg"
    elif mode=="loo":
        pred_path = "predictions_loo"
    elif mode=="mask":
        pred_path = "predictions_masked"  
        ths=["masked"]      
    else:
        pred_path="predictions"
        ths=["full"]
        
    dir_path = "{}/{}".format(gv.mg_model_path,pred_path)
    create_dir_if_not_exist(dir_path)
    
    pcc_results = DatasetMetadataSCV("{}/pcc_resuls.csv".format(dir_path))
    pcc_results.create_header(ths)
    mask_results = DatasetMetadataSCV("{}/mask_size_resuls.csv".format(dir_path))
    mask_results.create_header(ths)
    
    for image_index in images:
        print("image index: ",image_index)
        pccs = []
        mask_sizes = []
        ## Create image dir
        create_dir_if_not_exist("{}/{}".format(dir_path,image_index))
        
        ## Preprocess images
        px_start=center_xy[0]-margin[0]-xy_step
        py_start=center_xy[1]-margin[1]-xy_step
        pz_start = 0
        px_end=center_xy[0]+margin[0]+xy_step
        py_end=center_xy[1]+margin[1]+xy_step
        
        
        slice_by = [None,(px_start,px_end),(py_start,py_end)]
        print("preprocess..")
        input_image,target_image,target_seg_image,nuc_image,mem_image = preprocess_image(dataset,image_index,[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane"],slice_by=slice_by)
        pz_end = input_image.shape[0]
        
        weights = None
        
        ## Collect patchs
        input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,input_image)
        
        ## Batch predict
        ## Predicte unet and mask
        print("batch predict...")
        unet_patchs_p = predict(model.unet,input_patchs,batch_size)#model.unet.predict(tf.convert_to_tensor(input_patchs),batch_size=batch_size)
        mask_patchs_p = predict(model,input_patchs,batch_size)#model.predict(tf.convert_to_tensor(input_patchs),batch_size=batch_size)
        
        ## Create noise vector
        normal_noise = tf.random.normal(tf.shape(mask_patchs_p),stddev=noise_scale,dtype=tf.float64)
        # normal_noise = tf.random.uniform(tf.shape(mask_patchs_p),maxval=noise_scale,dtype=tf.float64)
        
        ## Back to image 
        weights = _get_weights(input_patchs[0].shape)
        unet_p,mask_p_full,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,mask_patchs_p,np.ones_like(mask_patchs_p)],weights,input_image.shape)
        mask_p_full = mask_p_full/d
        
        for th in ths:
            print(th)
            print("mask predict ...")
            create_dir_if_not_exist("{}/{}/{}".format(dir_path,image_index,th))
            if mode=="mask" and mask_image is not None:
                mask_image_ndarray = ImageUtils.image_to_ndarray(ImageUtils.imread(mask_image))/255.
                mask_p_full = mask_p_full*(1.-mask_image_ndarray)
            elif mode=="agg":
                mask_p = tf.cast(tf.where(mask_p_full>th,1.0,0.0),tf.float64).numpy()
            elif mode=="loo":
                ## 1.0 no noise, 0.0 is noise
                mask_p = tf.cast(tf.where(tf.math.logical_and(mask_p_full>(th-ths_step),mask_p_full<=th),1.0,0.0),tf.float64).numpy()
            else:
                mask_p = mask_p_full
            
            ## Create noisy input and predict unet
            mask_patchs_p_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p)
            mask_noise_patchs = (normal_noise*(1-mask_patchs_p_term))
            input_patchs_p = (mask_patchs_p_term*input_patchs)+mask_noise_patchs
            unet_noise_patchs_p = predict(model.unet,input_patchs_p.numpy(),batch_size)#model.unet.predict(tf.convert_to_tensor(input_patchs_p),batch_size=batch_size)
    
            ## Back to image 
            input_p,unet_noise_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p],weights,input_image.shape)
            
            ## Save images
            print("saving mask images...")
            base_save = "{}/{}/{}/".format(dir_path,image_index,th)
            ImageUtils.imsave(mask_p,"{}/mask_{}.tiff".format(base_save,image_index))
            ImageUtils.imsave(input_p/d,"{}/noisy_input_{}.tiff".format(base_save,image_index))
            ImageUtils.imsave(unet_noise_p/d,"{}/noisy_unet_prediction_{}.tiff".format(base_save,image_index))  
            pcc = pearson_corr((unet_p/d)[:,:,:], (unet_noise_p/d)[:,:,:])
            mask_size = np.sum(mask_p,dtype=np.float64)
            pccs.append(pcc)
            mask_sizes.append(mask_size)
            print("pearson corr for image:{} is :{}, mask ratio:{}".format(image_index,pcc,mask_size))
        
        print("saving global images...")
        image_save = "{}/{}/".format(dir_path,image_index)
        ImageUtils.imsave(input_image,"{}/input_{}.tiff".format(image_save,image_index))
        ImageUtils.imsave(target_image,"{}/target_{}.tiff".format(image_save,image_index))
        ImageUtils.imsave(nuc_image,"{}/nuc_{}.tiff".format(image_save,image_index))
        ImageUtils.imsave(mem_image,"{}/mem_{}.tiff".format(image_save,image_index))  
        ImageUtils.imsave(unet_p/d,"{}/unet_prediction_{}.tiff".format(image_save,image_index))   
        pcc_results.add_row(pccs)
        mask_results.add_row(mask_sizes)
    pcc_results.create()
    mask_results.create()
    
def analyze_correlations(organelles):
    ## Create main dir and organelle subdir
    pred_path = "predictions_correlations"
        
    dir_path = "{}/{}".format(gv.mg_model_path,pred_path)
    create_dir_if_not_exist(dir_path)
    
    patch_width = gv.patch_size[1]
    patch_height = gv.patch_size[2]
    
    for organelle in organelles:
        print(organelle)
        dir_path_organelle = "{}/{}".format(dir_path,organelle)
        create_dir_if_not_exist(dir_path)
    
        corr_results = DatasetMetadataSCV("{}/corr_resuls.csv".format(dir_path_organelle))
        corr_results.create_header(["source_organelle_size","target_organelle_size","intersection_size","correlation_size","pcc","pcc_wo","intersection_ratio","correlation_ratio"])
        train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(organelle)
        dataset = DataGen(train_ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=0.9, augment=False)
        for image_index in images:
            print("image index: ",image_index)
            
            ## Create image dir
            create_dir_if_not_exist("{}/{}".format(dir_path_organelle,image_index))
            
            ## Preprocess images
            px_start=center_xy[0]-patch_width-xy_step
            py_start=center_xy[1]-patch_height-xy_step
            px_end=center_xy[0]+patch_width+xy_step
            py_end=center_xy[1]+patch_height+xy_step
            
            slice_by = [None,(px_start,px_end),(py_start,py_end)]
            input_image,target_image,target_seg_image,nuc_image,mem_image = preprocess_image(dataset,image_index,[dataset.input_col,dataset.target_col,"structure_seg","channel_dna","channel_membrane"],slice_by=slice_by,normalize=[True,True,False,True,True])

            weights = None

            ## Init predictions
            unet_p = np.zeros_like(input_image)
            mask_p = np.zeros_like(input_image)
            input_p = np.zeros_like(input_image)
            unet_noise_p = np.zeros_like(input_image)
            unet_noise_p_wo = np.zeros_like(input_image)
            intersection_p = np.zeros_like(input_image)
            correlation_p = np.zeros_like(input_image)
            d = np.zeros_like(input_image)+1e-4

            # Init indexes
            pz=0
            px = px_start
            py = py_start
            
            while pz<=input_image.shape[0]-gv.patch_size[0]:
                while px<=px_end-gv.patch_size[1]:
                    while py<=py_end-gv.patch_size[2]: 
                        
                        ## Slice patch from input       
                        px_start_patch = px-px_start
                        py_start_patch = py-py_start    
                        s = [(pz,pz+gv.patch_size[0]),(px_start_patch,px_start_patch+gv.patch_size[1]),(py_start_patch,py_start_patch+gv.patch_size[2])]
                        input_patch = slice_image(input_image,s)
                        seg_patch = slice_image(target_seg_image,s)
                        # target_patch = slice_image(target_image,s)
                        
                        if weights is None:
                            weights = _get_weights(input_patch.shape)
                        
                        ## Predicte unet and mask
                        unet_patch_p = model.unet(np.expand_dims(input_patch,axis=0))
                        mask_patch_p = model(np.expand_dims(input_patch,axis=0))
                        
                        ## Create noise vector
                        normal_noise = tf.random.normal(tf.shape(mask_patch_p),stddev=noise_scale,dtype=tf.float64)
                        # normal_noise = tf.random.uniform(tf.shape(mask_patch_p),maxval=noise_scale,dtype=tf.float64)
                        mask_patch_p_term = np.where(seg_patch>0,0.0,mask_patch_p)
                            
                        ## Create noisy input and predict unet
                        mask_noise_patch = (normal_noise*(1-mask_patch_p))
                        mask_noise_patch_wo = (normal_noise*(1-mask_patch_p_term))
                        input_patch_p = input_patch+mask_noise_patch
                        input_patch_p_wo = input_patch+mask_noise_patch_wo
                        unet_noise_patch_p = model.unet(input_patch_p)
                        unet_noise_patch_p_wo = model.unet(input_patch_p_wo)
                        
                        ## Update with patchs
                        patch_slice = (slice(pz,pz+gv.patch_size[0]), slice(px_start_patch,px_start_patch+gv.patch_size[1]),slice(py_start_patch,py_start_patch+gv.patch_size[2]))
                        
                        unet_p[patch_slice] += unet_patch_p*weights 
                        mask_p[patch_slice] += mask_patch_p*weights #mask_patch_p*weights 
                        input_p[patch_slice] += input_patch_p*weights 
                        unet_noise_p[patch_slice] += unet_noise_patch_p*weights 
                        unet_noise_p_wo[patch_slice] += unet_noise_patch_p_wo*weights 
                        d[patch_slice] += weights[0]
                        
                        py+=xy_step
                    py=py_start  
                    px+=xy_step
                px=px_start
                pz+=z_step
            
            normalized_unet_p = ImageUtils.normalize(unet_p/d,1.0,np.float64)
            normalized_unet_p = np.where(normalized_unet_p>0.1, normalized_unet_p, 0.0)
            
            normalized_mask_p = np.where((mask_p/d)>0.5, (mask_p/d), 0.0)
            
            target_seg_image = target_seg_image/255.
            
            target_image = ImageUtils.normalize(target_image,1.0,np.float64)
            target_image = np.where(target_image>0.1, target_image, 0.0)
            
            intersection_p= (target_seg_image*normalized_unet_p) ## intersection of the predictions with the target organelle  
            correlation_p = (target_seg_image*normalized_mask_p) ## correlation of the attention predictions with the target organelle  
            ## Save images
            image_save = "{}/{}/".format(dir_path_organelle,image_index)
            ImageUtils.imsave(input_image,"{}/input_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(target_image,"{}/target_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(target_seg_image,"{}/seg_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(nuc_image,"{}/nuc_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(mem_image,"{}/mem_{}.tiff".format(image_save,image_index))  
            ImageUtils.imsave(normalized_unet_p,"{}/unet_prediction_{}.tiff".format(image_save,image_index))  
            ImageUtils.imsave(normalized_mask_p,"{}/mask_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(input_p/d,"{}/noisy_input_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(unet_noise_p/d,"{}/noisy_unet_prediction_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(unet_noise_p_wo/d,"{}/noisy_unet_prediction_wo_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(intersection_p,"{}/intersection_prediction_{}.tiff".format(image_save,image_index))
            ImageUtils.imsave(correlation_p,"{}/correlation_prediction_{}.tiff".format(image_save,image_index))
            
            pcc = pearson_corr((unet_p/d)[:,:,:], (unet_noise_p/d)[:,:,:])
            pcc_wo = pearson_corr((unet_p/d)[:,:,:], (unet_noise_p_wo/d)[:,:,:])
            
            corr_results.set_item(image_index,"source_organelle_size",np.sum(target_seg_image/np.prod(target_seg_image.shape),dtype=np.float64))
            corr_results.set_item(image_index,"target_organelle_size",np.sum(normalized_unet_p,dtype=np.float64))
            corr_results.set_item(image_index,"intersection_size",np.sum(intersection_p,dtype=np.float64))
            corr_results.set_item(image_index,"correlation_size",np.sum(correlation_p,dtype=np.float64))
            corr_results.set_item(image_index,"pcc",pcc)
            corr_results.set_item(image_index,"pcc_wo",pcc_wo)
            corr_results.set_item(image_index,"intersection_ratio",np.sum(intersection_p,dtype=np.float64)/np.sum(normalized_unet_p,dtype=np.float64))
            corr_results.set_item(image_index,"correlation_ratio",np.sum(correlation_p,dtype=np.float64)/np.sum(normalized_mask_p,dtype=np.float64))
            
        corr_results.create()

analyze_th("regular",mask)    
# analyze_th("agg")
# analyze_th("loo")
# analyze_correlations(organelles=["Nuclear-envelope","Endoplasmic-reticulum","Plasma-membrane","Desmosomes","Golgi","Microtubules","Actin-filaments","Nucleolus-(Dense-Fibrillar-Component)","Mitochondria","Endoplasmic-reticulum","Tight-junctions","Nucleolus-(Granular-Component)","Actomyosin-bundles"])
            
                
        
        
        
        
        
        
        

                

