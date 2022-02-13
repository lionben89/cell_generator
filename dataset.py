from copy import deepcopy
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

class PatchDataGen(keras.utils.Sequence):
    
    def __init__(self, image_list_csv, input_col, target_col,
                 batch_size,
                 input_as_y = False,
                 patch_size=(16,64,64,1),
                 mask = False,
                 mask_col = 'membrane_seg',
                 norm = True,
                 dilate = False,
                 dilate_kernel = np.ones((17,17),np.uint8)):
        
        self.df = DatasetMetadataSCV(image_list_csv,image_list_csv)
        self.n = self.df.get_shape()[0]
        self.input_col = input_col
        self.target_col = target_col
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.input_as_y = input_as_y ## output of target and input in Y
        self.mask = mask
        self.mask_col = mask_col
        self.norm = norm
        self.dilate = dilate
        self.dilate_kernel = dilate_kernel

        
        self.patches_from_image = 32
        self.buffer_size = self.batch_size*self.patches_from_image
        if (self.patch_size[0]==1): ## 2D image
            two_d_patch_size = self.patch_size[1:]
            self.input_buffer = np.zeros((self.buffer_size,*two_d_patch_size))
            self.target_buffer = np.zeros((self.buffer_size,*two_d_patch_size))
        else:
            self.input_buffer = np.zeros((self.buffer_size,*self.patch_size))
            self.target_buffer = np.zeros((self.buffer_size,*self.patch_size))
        self.fill_buffer()
        
        
    
    def on_epoch_end(self):
        self.fill_buffer()
    
    def fill_patches(self,i):
        image_index = np.random.randint(self.n)
        
        image_path = self.df.get_item(image_index,'path_tiff')
        image = None
        
        # print("adding image:{} to dataset cache\n".format(image_index))
        image = ImageUtils.imread(image_path)
        mask_image = None
        if (self.mask):
            mask_index = int(self.df.get_item(image_index,self.mask_col))
            mask_image = image.get_image_data("ZYXC")[ :, :,:,mask_index:mask_index+1]
            if (self.dilate):
                for i in range(mask_image.shape[1]):
                    cv2.dilate(mask_image[0,i],self.dilate_kernel,mask_image[0,i])   
                            
        channel_index = int(self.df.get_item(image_index,self.input_col))
        input_image = image.get_image_data("ZYXC")[ :, :,:,channel_index:channel_index+1]
        if (self.norm):
            input_image = ImageUtils.normalize(input_image,max_value=1.0,dtype=np.float32)             
        if (self.mask):
            input_image = mask_image(input_image,mask_image)
        input_image = ImageUtils.to_shape(input_image,input_image.shape,None, min_shape=self.patch_size)
        
        
        if (self.target_col == self.input_col):
            target_image = input_image
        else:
            channel_index = int(self.df.get_item(image_index,self.target_col))
            target_image = image.get_image_data("ZYXC")[ :, :,:,channel_index:channel_index+1]
            if (self.norm):
                target_image = ImageUtils.normalize(target_image,max_value=1.0,dtype=np.float32) 
            if (self.mask):
                target_image = mask_image(target_image,mask_image)
            target_image = ImageUtils.to_shape(target_image,input_image.shape,None, min_shape=self.patch_size)                 
        
        for j in range(self.patches_from_image):
            patch_indexes = []
            for k in range(len(self.patch_size)-1):
                random_patch_index = np.random.randint(input_image.shape[k]-self.patch_size[k]+1)
                patch_indexes.append((random_patch_index,random_patch_index+self.patch_size[k]))
            input_patch = slice_image(input_image,[*patch_indexes,(None,None)])
            target_patch = slice_image(target_image,[*patch_indexes,(None,None)])
            if (self.patch_size[0]==1): ## 2D image
                input_patch = input_patch[0]
                target_patch = target_patch[0]
            self.input_buffer[i*self.patches_from_image+j] = input_patch
            self.target_buffer[i*self.patches_from_image+j] = target_patch
    
    def fill_buffer(self):
        threads = []
        num_threads = self.patches_from_image
        for i in tqdm(range(self.batch_size)):
            if (num_threads > 1):
                thread = threading.Thread(target=self.fill_patches,args=[i])
                thread.start()
                threads.append(thread)
                
                if ((i+1)%num_threads == 0):
                    for t in (range(len(threads))):
                        threads[t].join()
                    threads = []
            else:
               self.fill_patches(i) 
                
        for thread in threads:
            thread.join()
        sklearn.utils.shuffle(self.input_buffer)
        sklearn.utils.shuffle(self.target_buffer)
           
    def __getitem__(self, index):
        X = self.input_buffer[index*self.batch_size:(index+1)*self.batch_size]
        # y1 = deepcopy(self.input_buffer[index*self.batch_size:(index+1)*self.batch_size])
        Y = self.target_buffer[index*self.batch_size:(index+1)*self.batch_size]
        if (self.input_as_y):
            Y = tuple([X,Y])
            
        return X,Y
    
    def __len__(self):
        return self.buffer_size // self.batch_size