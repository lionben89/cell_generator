import shutil
import os
import threading
import tensorflow.keras as keras
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from cell_imaging_utils.image.image_utils import ImageUtils
import numpy as np
from tqdm import tqdm
import sklearn as sklearn
import cv2

def slice_image(image_ndarray:np.ndarray, indexes:list)->np.ndarray:
    n_dim = len(image_ndarray.shape)
    slices = [slice(None)] * n_dim
    for i in range(len(indexes)):
        slices[i] = slice(indexes[i][0],indexes[i][1])
    slices = tuple(slices)
    sliced_image = image_ndarray[slices]
    return sliced_image

def mask_image_func(image_ndarray,mask_template_ndarray) -> np.ndarray:
    mask_ndarray = mask_template_ndarray
    return np.where(mask_ndarray==255,image_ndarray,np.zeros(image_ndarray.shape))

class DataGen(keras.utils.Sequence):
    
    def __init__(self, image_list_csv, input_col, target_col,
                 batch_size,
                 num_batches = 32,
                 patches_from_image = 1,
                 patch_size = (16,256,256,1),
                 input_as_y = False,
                 crop_edge = True,
                 mask = True,
                 mask_col = 'membrane_seg',
                 norm = True,
                 dilate = True,
                 dilate_kernel = np.ones((17,17),np.uint8)):
        
        self.new_path_origin = "/scratch/lionb@auth.ad.bgu.ac.il/{}/temp".format(os.environ.get('SLURM_JOB_ID'))
        self.df = DatasetMetadataSCV(image_list_csv,image_list_csv)
        self.n = self.df.get_shape()[0]
        self.input_col = input_col
        self.target_col = target_col
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.patches_from_image = patches_from_image ## if 1 resize image to patch size if >1 then sample patches with patch size
        self.patch_size = patch_size
        self.input_as_y = input_as_y ## output of target and input in Y
        self.crop_edge = crop_edge
        self.mask = mask
        self.mask_col = mask_col
        self.norm = norm
        self.dilate = dilate
        self.dilate_kernel = dilate_kernel

        
        self.patches_from_image = patches_from_image
        self.buffer_size = self.num_batches*self.batch_size*self.patches_from_image
        if (self.patch_size[0]==1): ## 2D image
            two_d_patch_size = self.patch_size[1:]
            self.input_buffer = np.zeros((self.buffer_size,*two_d_patch_size))
            self.target_buffer = np.zeros((self.buffer_size,*two_d_patch_size))
        else:
            self.input_buffer = np.zeros((self.buffer_size,*self.patch_size))
            self.target_buffer = np.zeros((self.buffer_size,*self.patch_size))
        
        if (not os.path.exists(self.new_path_origin)):
            os.makedirs(self.new_path_origin)
            
        self.fill_buffer()
        
    def get_image_from_ssd(self,file_path,prefix):
        file_name = file_path.split('/')[-1]    
        new_file_path = "{}/{}_{}".format(self.new_path_origin,prefix,file_name)
        if os.path.exists(new_file_path):
            image_ndarray = ImageUtils.image_to_ndarray(ImageUtils.imread(new_file_path))
            return image_ndarray, new_file_path
        return None, new_file_path    
    
    def on_epoch_end(self):
        self.fill_buffer()
    
    def fill_samples(self,i):
        for j in range(self.batch_size):
            image_index = np.random.randint(self.n)
            
            image_path = self.df.get_item(image_index,'path_tiff')
           
            input_image, input_new_file_path = self.get_image_from_ssd(image_path,self.input_col)
            target_image, target_new_file_path = self.get_image_from_ssd(image_path,self.target_col)
            if (input_image is None or target_image is None):
                image_ndarray = None
                
                # print("adding image:{} to dataset cache\n".format(image_index))
                image_ndarray = ImageUtils.image_to_ndarray(ImageUtils.imread(image_path))
                
                if (self.crop_edge):
                    mask_index = int(self.df.get_item(image_index,self.mask_col))
                    mask_template = ImageUtils.get_channel(image_ndarray,mask_index)
                    image_ndarray = ImageUtils.crop_edges(image_ndarray,mask_template)
                    
                mask_image = None
                if (self.mask):
                    mask_index = int(self.df.get_item(image_index,self.mask_col))
                    mask_image = ImageUtils.get_channel(image_ndarray,mask_index)
                    if (self.dilate):
                        for h in range(mask_image.shape[0]):
                            cv2.dilate(mask_image[0,h,:,:],self.dilate_kernel,mask_image[0,h,:,:])   

                channel_index = int(self.df.get_item(image_index,self.input_col))
                input_image = ImageUtils.get_channel(image_ndarray,channel_index) 
                if (self.norm):
                    input_image = ImageUtils.normalize(input_image,max_value=1.0,dtype=np.float32)             
                if (self.mask):
                    input_image = mask_image_func(input_image,mask_image)
                
                
                if (self.target_col == self.input_col):
                    target_image = input_image
                else:
                    channel_index = int(self.df.get_item(image_index,self.target_col))
                    target_image = ImageUtils.get_channel(image_ndarray,channel_index)
                    if (self.norm):
                        target_image = ImageUtils.normalize(target_image,max_value=1.0,dtype=np.float32)             
                    if (self.mask):
                        target_image = mask_image_func(target_image,mask_image)

                input_image = np.expand_dims(input_image[0],axis=-1)
                target_image = np.expand_dims(target_image[0],axis=-1)  
                ImageUtils.imsave(input_image,input_new_file_path)                   
                ImageUtils.imsave(target_image,target_new_file_path)                   
            if (self.patches_from_image > 1):
                #Sample patches 
                input_image = ImageUtils.to_shape(input_image,input_image.shape,None, min_shape=self.patch_size)
                target_image = ImageUtils.to_shape(target_image,input_image.shape,None, min_shape=self.patch_size)
                for k in range(self.patches_from_image):
                    patch_indexes = []
                    for o in range(len(self.patch_size)-1):
                        random_patch_index = np.random.randint(input_image.shape[o]-self.patch_size[o]+1)
                        patch_indexes.append((random_patch_index,random_patch_index+self.patch_size[o]))
                    input_patch = slice_image(input_image,[*patch_indexes,(None,None)])
                    target_patch = slice_image(target_image,[*patch_indexes,(None,None)])
                    if (self.patch_size[0]==1): ## 2D image
                        input_patch = input_patch[0]
                        target_patch = target_patch[0]
                    self.input_buffer[i*self.batch_size+j*self.patches_from_image+k] = input_patch
                    self.target_buffer[i*self.batch_size+j*self.patches_from_image+k] = target_patch
            else:
                # Resize to patch size
                # input_image = ndimage.zoom(input_image,[self.patch_size[0]/input_image.shape[0],1,1,1],mode="constant",cval=0,order=1,prefilter=False)
                # target_image = ndimage.zoom(target_image,[self.patch_size[0]/target_image.shape[0],1,1,1],mode="constant",cval=0,order=1,prefilter=False)   
                input_image = ImageUtils.to_shape(input_image,self.patch_size)
                target_image = ImageUtils.to_shape(target_image,self.patch_size)
                self.input_buffer[i*self.batch_size+j*self.patches_from_image] = input_image
                self.target_buffer[i*self.batch_size+j*self.patches_from_image] = target_image
            
            # if (old_image_path != image_path):
            #     os.remove(image_path)
                
    def fill_buffer(self):
        threads = []
        num_threads = 6
        for i in tqdm(range(self.num_batches)):
            if (num_threads > 1):
                thread = threading.Thread(target=self.fill_samples,args=[i])
                thread.start()
                threads.append(thread)
                
                if ((i+1)%num_threads == 0):
                    for t in (range(len(threads))):
                        threads[t].join()
                    threads = []
            else:
               self.fill_samples(i) 
                
        for thread in threads:
            thread.join()
        self.input_buffer,self.target_buffer = sklearn.utils.shuffle(self.input_buffer, self.target_buffer)
           
    def __getitem__(self, index):
        num_samples = int(self.buffer_size/self.num_batches)
        X = self.input_buffer[index*num_samples:(index+1)*num_samples]
        # y1 = deepcopy(self.input_buffer[index*self.batch_size:(index+1)*self.batch_size])
        Y = self.target_buffer[index*num_samples:(index+1)*num_samples]
        if (self.input_as_y):
            Y = tuple([X,Y])
            
        return X,Y
    
    def __len__(self):
        return self.num_batches