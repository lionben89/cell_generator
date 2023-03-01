from copy import deepcopy
from email.mime import image
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
import matplotlib.pyplot as plt

# tf.compat.v1.disable_eager_execution()


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

gv.mg_model_path = "mg_model_membrane_10_06_22_5_0_new"
gv.organelle = "Plasma-membrane" #"Golgi" #"Tight-junctions" #"Microtubules" #"Endoplasmic-reticulum" #"Plasma-membrane" 
#"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)","Actin-filaments"
gv.train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_test.csv".format(gv.organelle)

compound = None #"staurosporine"#"paclitaxol_vehicle" #"rapamycin" #"paclitaxol" #"blebbistatin" #"staurosporine"
if compound is not None:
    ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}_{}/image_list_test.csv".format(gv.organelle,compound)
else:
    ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(gv.organelle)

# dataset = DataGen(gv.train_ds_path, gv.input, gv.target, batch_size=1, num_batches=1, patch_size=gv.patch_size, min_precentage=0, max_precentage=1, augment=False)
dataset = DataGen(ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)

## Choose images
images = [7]#range(5,9,1) #list(np.random.randint(0,dataset.df.get_shape()[0],30,dtype=int))#range(dataset.df.get_shape()[0]) #list(np.random.randint(0,dataset.df.get_shape()[0],30,dtype=int))#range(dataset.df.get_shape()[0])

## Noise
noise_scale = 5.0

## Batch size
batch_size=1

## Image center
center_xy = [312,462] #[200,100]
margin=[192,256] #[256,256]#

## Patch steps
xy_step = 64
z_step = 16

## Load model
print("Loading model:",gv.mg_model_path)
model = keras.models.load_model(gv.mg_model_path)

def predict(model,data,batch_size):   
    tf.keras.backend.clear_session()
    _ = gc.collect() 
    # batch_data = data.reshape((-1,batch_size,*gv.patch_size))
    # output = np.zeros_like(batch_data)
    # for i in range(batch_data.shape[0]):
    #     batch_pred = model.predict_on_batch(batch_data[i])
    #     output[i] = batch_pred
    # output = output.reshape(data.shape)
    
    output = model.predict(data,batch_size=batch_size)
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
                
                py+=min(xy_step,max(1,py_end-gv.patch_size[2]-py))
            py=py_start  
            px+=min(xy_step,max(1,px_end-gv.patch_size[1]-px))
        px=px_start
        pz+=min(z_step,max(1,pz_end-gv.patch_size[0]-pz))
        
    # if pz+z_step < pz_end:
        # pz = pz_end - gv.patch_size[0]
        # while px<=px_end-gv.patch_size[1]:
        #     while py<=py_end-gv.patch_size[2]: 
                
        #         ## Slice patch from input       
        #         px_start_patch = px-px_start
        #         py_start_patch = py-py_start    
        #         s = [(pz,pz+gv.patch_size[0]),(px_start_patch,px_start_patch+gv.patch_size[1]),(py_start_patch,py_start_patch+gv.patch_size[2])]
        #         patch = slice_image(image,s)
        #         # seg_patch = slice_image(target_seg_image,s)
                
        #         patchs.append(patch)
                
        #         py+=xy_step
        #     py=py_start  
        #     px+=xy_step
    return np.array(patchs)

def assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,patchs,weights,assembled_image_shape):
    print("assemble images from patchs...")
    patchs = np.array(patchs)
    assembled_images = np.zeros((patchs.shape[0],*assembled_image_shape))
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
                
                py+=min(xy_step,max(1,py_end-gv.patch_size[2]-py))
                i+=1
                
            py=py_start  
            px+=min(xy_step,max(1,px_end-gv.patch_size[1]-px))
        px=px_start
        pz+=min(z_step,max(1,pz_end-gv.patch_size[0]-pz))     
    return assembled_images

def analyze_th(mode,mask_image=None,manual_th="full",save_image=True,save_histo=False):
    ## Create thresholds
    num_bins = 100
    ths_start = 0.65
    ths_step = 0.05
    ths_stop = 1.05
    ths = np.arange(ths_start,ths_stop,ths_step)
    ## Create main dir
    if mode=="agg":
        pred_path = "predictions_agg"
    elif mode=="loo":
        pred_path = "predictions_loo"
    elif mode=="mask":
        pred_path = "predictions_masked/gc"  
        # ths = [0.0]
        ths=[0.0,0.000075,0.0001,0.0002,0.0005,0.0008,0.001]
    elif mode=="regular":
        pred_path="predictions"
        ths=[manual_th]
    if compound is not None:
        pred_path = "{}_{}".format(pred_path,compound)
        
    dir_path = "{}/{}".format(gv.mg_model_path,pred_path)
    create_dir_if_not_exist(dir_path)
    
    pcc_results = DatasetMetadataSCV("{}/pcc_resuls.csv".format(dir_path))
    pcc_results.create_header(ths)
    mask_results = DatasetMetadataSCV("{}/mask_size_resuls.csv".format(dir_path))
    mask_results.create_header(ths)
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
                
                ## mask_p = (1. - mask_image_ndarray*0.25)
                mask_p = tf.cast(tf.where(mask_image_ndarray>=th,1.0,0.0),tf.float64).numpy()
                mask_p_binary = tf.cast(tf.where(mask_image_ndarray>=th,1.0,0.0),tf.float64).numpy()
            elif mode=="agg":
                mask_p_binary = tf.cast(tf.where(mask_p_full/d>th,1.0,0.0),tf.float64).numpy()
                mask_p = tf.cast(tf.where(mask_p_full/d>th,1.0,0.5),tf.float64).numpy()
            elif mode=="loo":
                ## 1.0 no noise, 0.0 is noise
                mask_p_binary = tf.cast(tf.where(tf.math.logical_and(mask_p_full/d>(th-ths_step),mask_p_full/d<=th),1.0,0.0),tf.float64).numpy()
                mask_p = tf.cast(tf.where(tf.math.logical_and(mask_p_full/d>(th-ths_step),mask_p_full/d<=th),1.0,0.0),tf.float64).numpy()
            else:
                if manual_th != "full":
                    mask_p_binary = tf.cast(tf.where(mask_p_full/d>th,1.0,0.0),tf.float64).numpy()
                    mask_p = tf.cast(tf.where(mask_p_full/d>th,1.0,0.0),tf.float64).numpy()
                else:
                    mask_p_binary = np.ones_like(mask_p_full)
                    mask_p = mask_p_full/d
            mask_size = np.sum(mask_p_binary,dtype=np.float64)
            mask_organelle_intersection = np.sum(mask_p_binary*target_seg_image,dtype=np.float64)/mask_size
            mask_size = mask_size/np.prod(mask_p_binary.shape)
            
            ## Create noisy input and predict unet
            mask_patchs_p_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p)
            mask_noise_patchs = (normal_noise*(1-mask_patchs_p_term))
            input_patchs_p = (mask_patchs_p_term*input_patchs)+mask_noise_patchs
            unet_noise_patchs_p = predict(model.unet,input_patchs_p.numpy(),batch_size)#model.unet.predict(tf.convert_to_tensor(input_patchs_p),batch_size=batch_size)
    
            ## Back to image 
            input_p,unet_noise_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p],weights,input_image.shape)
            
            ## Save images
            if save_image:
                print("saving mask images...")
                base_save = "{}/{}/{}/".format(dir_path,image_index,th)
                ImageUtils.imsave((mask_p).astype(np.float16),"{}/mask_{}.tiff".format(base_save,image_index))
                ImageUtils.imsave((input_p/d).astype(np.float16),"{}/noisy_input_{}.tiff".format(base_save,image_index))
                ImageUtils.imsave((unet_noise_p/d).astype(np.float16),"{}/noisy_unet_prediction_{}.tiff".format(base_save,image_index))  
            # pcc = pearson_corr((unet_p*mem_seg_image/d)[:,:,:], (unet_noise_p*mem_seg_image/d)[:,:,:])
            pcc = pearson_corr((unet_p/d)[:,:,:], (unet_noise_p/d)[:,:,:])
            pccs.append(pcc)
            mask_sizes.append(mask_size)
            c = 1/(mask_organelle_intersection/mask_size)
            context.append(c)
            print("pearson corr for image:{} is :{}, mask ratio:{} context:{}".format(image_index,pcc,mask_size,c))
                       
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
            h, _ = np.histogram((mask_p).reshape(1,-1), bins=num_bins, density=True,range=(0.01,1.01)) #ignore 0.0 for single cells
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
    
def analyze_correlations(organelles,mask_th,save_images=True,organelle_precent_pixels=1.0):
    ## Create main dir and organelle subdir
    pred_path = "predictions_correlations"
        
    base_dir_path = "{}/{}".format(gv.mg_model_path,pred_path)
    create_dir_if_not_exist(base_dir_path)
    repeat = 5
    for organelle in organelles:
        print(organelle)
        dir_path = "{}/{}".format(base_dir_path,organelle)
        create_dir_if_not_exist(dir_path)
    
        corr_results = DatasetMetadataSCV("{}/corr_resuls_{}_organelle_precent_pixels_{}.csv".format(dir_path,organelle,organelle_precent_pixels))
        corr_results.create_header(["ratio_of_organelle_in_image","ratio_of_mask_in_image","ratio_of_mask_organelle_intersection_in_mask","ratio_of_mask_organelle_intersection_in_organelle","ratio_of_random_pixels_in_mask","ratio_of_mask_organelle_intersection_in_image","ratio_of_random_pixels_organelle_in_mask","ratio_of_random_pixels_outside_organelle_in_mask","pcc","pcc_wo","pcc_wo_organelle_pixels","pcc_random"])
        ds_organelle = organelle
        train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(ds_organelle)
        test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_test.csv".format(ds_organelle)
        dataset = DataGen(train_ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
        images = range(min([dataset.df.get_shape()[0],10]))
        item = 0
        for image_index in images:
            for r in range(repeat):
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
                input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,input_image)
                
                ## Batch predict
                ## Predicte unet and mask
                print("batch predict...")
                unet_patchs_p = predict(model.unet,input_patchs,batch_size)
                mask_patchs_p = predict(model,input_patchs,batch_size)
                
                ## Create noise vector
                normal_noise = tf.random.normal(tf.shape(mask_patchs_p),stddev=noise_scale,dtype=tf.float32)
                # normal_noise = tf.random.uniform(tf.shape(mask_patchs_p),maxval=noise_scale,dtype=tf.float64)
                
                ## Back to image 
                weights = _get_weights(input_patchs[0].shape)
                unet_p,mask_p_full,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,mask_patchs_p,np.ones_like(mask_patchs_p)],weights,input_image.shape)
                mask_p_full = mask_p_full/d
                
                print("mask predict ...")
                mask_p_binary = tf.where(mask_p_full>mask_th,1.0,0.0)
                mask_p_binary_wo = (mask_p_binary * (1-target_seg_image))
                mask_p = tf.where((mask_p_binary)>0.0,1.0,0.5)
                mask_p_wo = tf.where(mask_p_binary_wo>0.0,1.0,0.5)
                mask_organelle_itersection = (mask_p_binary * target_seg_image)
                ratio_of_organelle_in_image = np.sum(target_seg_image,dtype=np.float64)/np.prod(target_seg_image.shape)
                ratio_of_mask_organelle_intersection_in_image = np.sum(mask_organelle_itersection,dtype=np.float64)/np.prod(mask_organelle_itersection.shape)
                ratio_of_mask_in_image = np.sum(mask_p_binary,dtype=np.float64)/np.prod(mask_p_binary.shape)
                ratio_of_mask_organelle_intersection_in_mask = np.sum(mask_organelle_itersection,dtype=np.float64)/np.sum(mask_p_binary,dtype=np.float64)
                random_seg_image = np.random.choice([0.0,1.],target_seg_image.shape,p=[1-((ratio_of_mask_organelle_intersection_in_mask * organelle_precent_pixels) / (1-ratio_of_mask_organelle_intersection_in_mask)),((ratio_of_mask_organelle_intersection_in_mask * organelle_precent_pixels) / (1-ratio_of_mask_organelle_intersection_in_mask))])
                
                random_seg_image_wo_organele = np.random.choice([0.0,1.],target_seg_image.shape,p=[1-(ratio_of_mask_organelle_intersection_in_mask * organelle_precent_pixels),ratio_of_mask_organelle_intersection_in_mask * organelle_precent_pixels])
                mask_p_random_binary = (mask_p_binary * (1-random_seg_image))
                mask_p_random = tf.where(mask_p_random_binary>0.0,1.0,0.5)
                mask_p_random_outside_organelle_binary = tf.math.minimum((mask_p_binary * (1-random_seg_image_wo_organele))+(target_seg_image * mask_p_binary),1.0)
                mask_p_random_outside_organelle = tf.where(mask_p_random_outside_organelle_binary>0.0,1.0,0.5)
                
                organelle_pixels_target_seg_image = np.random.choice([0.0,1.],target_seg_image.shape,p=[1-(organelle_precent_pixels),organelle_precent_pixels])
                mask_p_binary_wo_organelle_pixels = tf.math.minimum((mask_p_binary * (1-organelle_pixels_target_seg_image))+((1-target_seg_image)*mask_p_binary),1.0)
                mask_p_wo_organelle_pixels = tf.where(mask_p_binary_wo_organelle_pixels>0.0,1.0,0.5)
                
                ## Create noisy input and predict unet
                mask_patchs_p_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p)
                mask_patchs_p_wo_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p_wo)
                mask_patchs_p_wo_organelle_pixels_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p_wo_organelle_pixels)
                mask_patchs_p_random_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p_random)
                mask_patchs_p_random_outside_organelle_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p_random_outside_organelle)
                # neg_organelle_patchs_p_wo_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,1-target_seg_image)
                
                mask_noise_patchs = (normal_noise*(1-mask_patchs_p_term))
                mask_noise_wo_patchs = (normal_noise*(1-mask_patchs_p_wo_term))
                mask_noise_wo_organelle_pixels_patchs = (normal_noise*(1-mask_patchs_p_wo_organelle_pixels_term))
                mask_noise_random_patchs = (normal_noise*(1-mask_patchs_p_random_term))
                mask_noise_random_outside_organelle_patchs = (normal_noise*(1-mask_patchs_p_random_outside_organelle_term))
                # mask_noise_neg_organelle_patchs = (normal_noise*(1-neg_organelle_patchs_p_wo_term))
                
                input_patchs_p = (mask_patchs_p_term*input_patchs)+mask_noise_patchs
                input_patchs_wo_p = (mask_patchs_p_wo_term*input_patchs)+mask_noise_wo_patchs
                input_patchs_wo_organelle_pixels_p = (mask_patchs_p_wo_organelle_pixels_term*input_patchs)+mask_noise_wo_organelle_pixels_patchs
                input_patchs_random_p = (mask_patchs_p_random_term*input_patchs)+mask_noise_random_patchs
                input_patchs_random_outside_organelle_p = (mask_patchs_p_random_term*input_patchs)+mask_noise_random_outside_organelle_patchs
                # input_patchs_neg_organelle_p = (mask_noise_neg_organelle_patchs*input_patchs)+mask_noise_neg_organelle_patchs
                
                unet_noise_patchs_p = predict(model.unet,input_patchs_p.numpy(),batch_size)
                unet_noise_patchs_wo_p = predict(model.unet,input_patchs_wo_p.numpy(),batch_size)
                unet_noise_patchs_wo_organelle_pixels_p = predict(model.unet,input_patchs_wo_organelle_pixels_p.numpy(),batch_size)
                unet_noise_patchs_random_p = predict(model.unet,input_patchs_random_p.numpy(),batch_size)
                unet_noise_patchs_random_outside_organelle_p = predict(model.unet,input_patchs_random_outside_organelle_p.numpy(),batch_size)
                # unet_noise_patchs_neg_organelle_p = predict(model.unet,input_patchs_neg_organelle_p.numpy(),batch_size)
        
                ## Back to image 
                input_p,unet_noise_p,unet_noise_wo_p,unet_noise_random_p,unet_noise_wo_organelle_pixels_p,unet_noise_random_outside_organelle_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p,unet_noise_patchs_wo_p,unet_noise_patchs_random_p,unet_noise_patchs_wo_organelle_pixels_p,unet_noise_patchs_random_outside_organelle_p],weights,input_image.shape)
                
                ## Save images
                base_save = "{}/{}/".format(dir_path,image_index)
                if image_index<save_images:
                    ## Create image dir
                    create_dir_if_not_exist(base_save)
                    print("saving mask images...")
                    ImageUtils.imsave(mask_p_binary.numpy().astype(np.float16),"{}/mask_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave(mask_p_binary_wo.numpy().astype(np.float16),"{}/mask_wo_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave(mask_p_random_binary.numpy().astype(np.float16),"{}/mask_random_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave(mask_p_random_outside_organelle_binary.numpy().astype(np.float16),"{}/mask_random_outside_organelle_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave(mask_p_binary_wo_organelle_pixels.numpy().astype(np.float16),"{}/mask_wo_organelle_pixels_{}.tiff".format(base_save,image_index))            
                    ImageUtils.imsave(target_seg_image.astype(np.float16),"{}/seg_source_organelle_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((input_p/d).astype(np.float16),"{}/noisy_input_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((unet_noise_p/d).astype(np.float16),"{}/noisy_unet_prediction_{}.tiff".format(base_save,image_index)) 
                    ImageUtils.imsave((unet_noise_wo_p/d).astype(np.float16),"{}/noisy_unet_prediction_wo_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((unet_noise_random_p/d).astype(np.float16),"{}/noisy_unet_prediction_random_{}.tiff".format(base_save,image_index))  
                    ImageUtils.imsave((unet_noise_random_outside_organelle_p/d).astype(np.float16),"{}/noisy_unet_prediction_random_outside_organelle_{}.tiff".format(base_save,image_index))  
                    ImageUtils.imsave((unet_noise_wo_organelle_pixels_p).astype(np.float16),"{}/noisy_unet_prediction_wo_organelle_pixels_{}.tiff".format(base_save,image_index))
                    # ImageUtils.imsave(unet_noise_neg_organelle_p/d,"{}/noisy_unet_prediction_neg_organelle_{}.tiff".format(base_save,image_index))
                pcc = pearson_corr((unet_p/d)[:,:,:], (unet_noise_p/d)[:,:,:])
                pcc_wo = pearson_corr((unet_p/d)[:,:,:], (unet_noise_wo_p/d)[:,:,:])
                pcc_wo_organelle_pixels = pearson_corr((unet_p/d)[:,:,:], (unet_noise_wo_organelle_pixels_p/d)[:,:,:])
                # pcc_neg_organelle = pearson_corr((unet_p/d)[:,:,:], (unet_noise_neg_organelle_p/d)[:,:,:])
                pcc_random = pearson_corr((unet_p/d)[:,:,:], (unet_noise_random_p/d)[:,:,:])
                pcc_random_outside_organelle = pearson_corr((unet_p/d)[:,:,:], (unet_noise_random_outside_organelle_p/d)[:,:,:])
                
                print("pearson corr for image:{} is :{} wo organelle is:{} random occlusion is:{} wo organelle pixels:{} random occlusion outside organelle:{}".format(image_index,pcc,pcc_wo,pcc_random,pcc_wo_organelle_pixels,pcc_random_outside_organelle))
                
                if image_index<save_images:
                    print("saving global images...")
                    ImageUtils.imsave(input_image.astype(np.float16),"{}/input_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave(target_image.astype(np.float16),"{}/target_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave(nuc_image.astype(np.float16),"{}/nuc_{}.tiff".format(base_save,image_index)) 
                    ImageUtils.imsave(mem_image.astype(np.float16),"{}/mem_{}.tiff".format(base_save,image_index))  
                    ImageUtils.imsave((unet_p/d).astype(np.float16),"{}/unet_prediction_{}.tiff".format(base_save,image_index))   
                
                corr_results.set_item(item,"ratio_of_organelle_in_image",ratio_of_organelle_in_image)
                corr_results.set_item(item,"ratio_of_mask_in_image",ratio_of_mask_in_image)
                corr_results.set_item(item,"ratio_of_mask_organelle_intersection_in_image",ratio_of_mask_organelle_intersection_in_image)
                corr_results.set_item(item,"ratio_of_mask_organelle_intersection_in_mask",ratio_of_mask_organelle_intersection_in_mask)
                corr_results.set_item(item,"ratio_of_mask_organelle_intersection_in_organelle",np.sum(mask_organelle_itersection,dtype=np.float64)/np.sum(target_seg_image,dtype=np.float64))
                corr_results.set_item(item,"ratio_of_random_pixels_in_mask",1 - np.sum(mask_p_random_binary.numpy(),dtype=np.float64)/np.sum(mask_p_binary,dtype=np.float64))
                corr_results.set_item(item,"ratio_of_random_pixels_organelle_in_mask",1 - np.sum(mask_p_binary_wo_organelle_pixels.numpy(),dtype=np.float64)/np.sum(mask_p_binary,dtype=np.float64))
                corr_results.set_item(item,"ratio_of_random_pixels_outside_organelle_in_mask",1 - np.sum((mask_p_random_outside_organelle_binary.numpy()),dtype=np.float64)/np.sum(mask_p_binary,dtype=np.float64))
        
                corr_results.set_item(item,"pcc",pcc)
                corr_results.set_item(item,"pcc_wo",pcc_wo)
                corr_results.set_item(item,"pcc_wo_organelle_pixels",pcc_wo_organelle_pixels)
                corr_results.set_item(item,"pcc_random",pcc_random)
                corr_results.set_item(item,"pcc_random_outside_organelle",pcc_random_outside_organelle)
                # corr_results.set_item(image_index,"pcc_neg_organelle",pcc_neg_organelle)
                item +=1
                del input_patchs_p
                del input_patchs_random_p
                del input_patchs_wo_p
                del input_patchs_wo_organelle_pixels_p
                del normal_noise
                del unet_noise_patchs_p
                del unet_noise_patchs_random_p
                del unet_noise_patchs_wo_p
                del unet_noise_wo_organelle_pixels_p
                del input_p
                del unet_noise_p
                del unet_noise_wo_p
                del unet_noise_random_p
                del mask_noise_patchs
                del mask_noise_wo_patchs  
                del mask_noise_wo_organelle_pixels_patchs
                del mask_noise_random_patchs
                del mask_p_random_outside_organelle
                del unet_noise_patchs_random_outside_organelle_p
                del input_patchs_random_outside_organelle_p
                del mask_patchs_p_random_outside_organelle_term
            
        corr_results.create()

def analyze_correlations_constant(organelles,mask_th,save_images=True,absoulte_organelle_precent_pixels=0.05):
    ## Create main dir and organelle subdir
    pred_path = "predictions_correlations_constant"
        
    base_dir_path = "{}/{}".format(gv.mg_model_path,pred_path)
    create_dir_if_not_exist(base_dir_path)
    repeat = 5
    for organelle in organelles:
        print(organelle)
        dir_path = "{}/{}".format(base_dir_path,organelle)
        create_dir_if_not_exist(dir_path)
    
        corr_results = DatasetMetadataSCV("{}/corr_resuls_{}_organelle_precent_pixels_{}.csv".format(dir_path,organelle,absoulte_organelle_precent_pixels))
        corr_results.create_header(["ratio_of_organelle_in_image","ratio_of_mask_in_image","ratio_of_mask_organelle_intersection_in_mask","ratio_of_mask_organelle_intersection_in_organelle","ratio_of_mask_organelle_intersection_in_image", "ratio_of_noised_pixels_in_mask","pcc","pcc_wo_organelle_pixels"])
        ds_organelle = organelle
        train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(ds_organelle)
        test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_test.csv".format(ds_organelle)
        dataset = DataGen(train_ds_path ,gv.input,gv.target,batch_size = 1, num_batches = 1, patch_size=gv.patch_size,min_precentage=0.0,max_precentage=1.0, augment=False)
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
            input_patchs = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,input_image)
            
            ## Batch predict
            ## Predicte unet and mask
            print("batch predict...")
            unet_patchs_p = predict(model.unet,input_patchs,batch_size)
            mask_patchs_p = predict(model,input_patchs,batch_size)
            
            ## Create noise vector
            normal_noise = tf.random.normal(tf.shape(mask_patchs_p),stddev=noise_scale,dtype=tf.float32)
            # normal_noise = tf.random.uniform(tf.shape(mask_patchs_p),maxval=noise_scale,dtype=tf.float64)
            
            ## Back to image 
            weights = _get_weights(input_patchs[0].shape)
            unet_p,mask_p_full,d = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[unet_patchs_p,mask_patchs_p,np.ones_like(mask_patchs_p)],weights,input_image.shape)
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
                print("mask predict ...")
                mask_p_binary = tf.where(mask_p_full>mask_th,1.0,0.0)
                mask_p = tf.where((mask_p_binary)>0.0,1.0,0.0)
                mask_organelle_itersection = (mask_p_binary * target_seg_image)
                ratio_of_organelle_in_image = np.sum(target_seg_image,dtype=np.float64)/np.prod(target_seg_image.shape)
                ratio_of_mask_organelle_intersection_in_image = np.sum(mask_organelle_itersection,dtype=np.float64)/np.prod(mask_organelle_itersection.shape)
                ratio_of_mask_in_image = np.sum(mask_p_binary,dtype=np.float64)/np.prod(mask_p_binary.shape)
                ratio_of_mask_organelle_intersection_in_mask = np.sum(mask_organelle_itersection,dtype=np.float64)/np.sum(mask_p_binary,dtype=np.float64)
                
                organelle_precent_pixels = absoulte_organelle_precent_pixels/ratio_of_mask_organelle_intersection_in_mask
                if organelle_precent_pixels > 1.0:
                    item +=1
                    continue
                organelle_pixels_target_seg_image = np.random.choice([0.0,1.],target_seg_image.shape,p=[1-(organelle_precent_pixels),organelle_precent_pixels])
                mask_p_binary_wo_organelle_pixels = tf.math.minimum((mask_p_binary * (1-organelle_pixels_target_seg_image))+((1-target_seg_image)*mask_p_binary),1.0)
                mask_p_wo_organelle_pixels = tf.where(mask_p_binary_wo_organelle_pixels>0.0,1.0,0.0)
                
                ## Create noisy input and predict unet
                mask_patchs_p_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p)
                mask_patchs_p_wo_organelle_pixels_term = collect_patchs(px_start,py_start,pz_start,px_end,py_end,pz_end,mask_p_wo_organelle_pixels)
                
                mask_noise_patchs = (normal_noise*(1-mask_patchs_p_term))
                mask_noise_wo_organelle_pixels_patchs = (normal_noise*(1-mask_patchs_p_wo_organelle_pixels_term))
                
                input_patchs_p = (mask_patchs_p_term*input_patchs)+mask_noise_patchs
                input_patchs_wo_organelle_pixels_p = (mask_patchs_p_wo_organelle_pixels_term*input_patchs)+mask_noise_wo_organelle_pixels_patchs
                
                unet_noise_patchs_p = predict(model.unet,input_patchs_p.numpy(),batch_size)
                unet_noise_patchs_wo_organelle_pixels_p = predict(model.unet,input_patchs_wo_organelle_pixels_p.numpy(),batch_size)
        
                ## Back to image 
                input_p,unet_noise_p,unet_noise_wo_organelle_pixels_p = assemble_image(px_start,py_start,pz_start,px_end,py_end,pz_end,[input_patchs_p,unet_noise_patchs_p,unet_noise_patchs_wo_organelle_pixels_p],weights,input_image.shape)
                
                ## Save images
                if image_index<save_images:
                    print("saving mask images...")
                    ImageUtils.imsave(mask_p_binary.numpy().astype(np.float16),"{}/mask_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave(mask_p_binary_wo_organelle_pixels.numpy().astype(np.float16),"{}/mask_wo_organelle_pixels_{}.tiff".format(base_save,image_index))            
                    ImageUtils.imsave(target_seg_image.astype(np.float16),"{}/seg_source_organelle_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((input_p/d).astype(np.float16),"{}/noisy_input_{}.tiff".format(base_save,image_index))
                    ImageUtils.imsave((unet_noise_p/d).astype(np.float16),"{}/noisy_unet_prediction_{}.tiff".format(base_save,image_index)) 
                    ImageUtils.imsave((unet_noise_wo_organelle_pixels_p).astype(np.float16),"{}/noisy_unet_prediction_wo_organelle_pixels_{}.tiff".format(base_save,image_index))

                pcc = pearson_corr((unet_p/d)[:,:,:], (unet_noise_p/d)[:,:,:])
                pcc_wo_organelle_pixels = pearson_corr((unet_p/d)[:,:,:], (unet_noise_wo_organelle_pixels_p/d)[:,:,:])

                
                print("pearson corr for image:{} reapet:{} is :{} wo organelle pixels:{}".format(image_index,r,pcc,pcc_wo_organelle_pixels)) 
                
                corr_results.set_item(item,"ratio_of_organelle_in_image",ratio_of_organelle_in_image)
                corr_results.set_item(item,"ratio_of_mask_in_image",ratio_of_mask_in_image)
                corr_results.set_item(item,"ratio_of_mask_organelle_intersection_in_image",ratio_of_mask_organelle_intersection_in_image)
                corr_results.set_item(item,"ratio_of_mask_organelle_intersection_in_mask",ratio_of_mask_organelle_intersection_in_mask)
                corr_results.set_item(item,"ratio_of_mask_organelle_intersection_in_organelle",np.sum(mask_organelle_itersection,dtype=np.float64)/np.sum(target_seg_image,dtype=np.float64))
                corr_results.set_item(item,"ratio_of_noised_pixels_in_mask",1 - np.sum(mask_p_binary_wo_organelle_pixels.numpy(),dtype=np.float64)/np.sum(mask_p_binary,dtype=np.float64))
                
        
                corr_results.set_item(item,"pcc",pcc)
                corr_results.set_item(item,"pcc_wo_organelle_pixels",pcc_wo_organelle_pixels)
                
                item +=1
                del input_patchs_p
                del input_patchs_wo_organelle_pixels_p               
                del unet_noise_patchs_p
                del unet_noise_wo_organelle_pixels_p
                del input_p
                del unet_noise_p
                del mask_noise_patchs
                del mask_noise_wo_organelle_pixels_patchs
                
            del normal_noise
            del unet_patchs_p
            del mask_patchs_p
        corr_results.create()
        
# analyze_correlations(save_images=3, mask_th=0.7, organelle_precent_pixels=0.5, organelles=["Plasma-membrane","Nuclear-envelope","Endoplasmic-reticulum","Nucleolus-(Dense-Fibrillar-Component)","Nucleolus-(Granular-Component)","Microtubules","Tight-junctions","Mitochondria","Actomyosin-bundles","Actin-filaments","Desmosomes","Golgi"])
# analyze_correlations_constant(save_images=3, mask_th=0.8, absoulte_organelle_precent_pixels=0.01, organelles=["Golgi","Plasma-membrane","Nuclear-envelope","Endoplasmic-reticulum","Nucleolus-(Dense-Fibrillar-Component)","Nucleolus-(Granular-Component)","Microtubules","Tight-junctions","Mitochondria","Actomyosin-bundles","Actin-filaments","Desmosomes"])
# analyze_correlations_constant(save_images=3, mask_th=0.5, absoulte_organelle_precent_pixels=0.01, organelles=["Lysosome","Adherens-junctions","Gap-junctions","Matrix-adhesions","Peroxisomes","Endosomes"])
# analyze_correlations_constant(save_images=3, mask_th=0.5, absoulte_organelle_precent_pixels=0.01, organelles=["Microtubules"])

# analyze_th("regular",save_histo=False,save_image=True)
# analyze_th(mode="agg",mask_image=None,manual_th="full",save_image=True,save_histo=False)
# for mth in [0.85]:
    # analyze_th("regular",mask_image=None,manual_th=mth,save_image=True,save_histo=False)
analyze_th("mask","{}/X_gradcam_layer_downsample_4_full.tiff".format(gv.mg_model_path)) #saliency_full.tiff #X_gradcam_layer_downsample_4_full.tiff
# analyze_th("mask","{}/predictions_agg/1/MASK_4.tif".format(gv.mg_model_path))
# analyze_th("mask","{}/predictions_masked/4/MASK_2.tif".format(gv.mg_model_path))
# analyze_th("agg")
# analyze_th("loo")
# analyze_correlations(save_images=3,mask_th=0.85,organelles=["Endoplasmic-reticulum","Plasma-membrane","Actin-filaments"])
# analyze_correlations(save_images=3,mask_th=0.85,organelles=["Golgi","Microtubules","Nucleolus-(Dense-Fibrillar-Component)"])
# analyze_correlations(save_images=3,mask_th=0.85,organelles=["Mitochondria","Tight-junctions","Nucleolus-(Granular-Component)"])
# analyze_correlations(save_images=3,mask_th=0.85,organelles=["Actomyosin-bundles","Nuclear-envelope","Desmosomes"])
                
        
        
        
        
        
        
        

                

