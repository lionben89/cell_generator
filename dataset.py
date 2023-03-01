import shutil
import os
import threading
import tensorflow as tf
import tensorflow.keras as keras
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from cell_imaging_utils.image.image_utils import ImageUtils
import numpy as np
from tqdm import tqdm
import sklearn as sklearn
import cv2
from patchify import patchify

def normalize(image_ndarray,max_value=255,dtype=np.uint8) -> np.ndarray:
    image_ndarray = image_ndarray.astype(np.float64)
    max_var = np.max(image_ndarray!=np.inf)
    image_ndarray = np.where(image_ndarray==np.inf,max_var,image_ndarray)
    temp_image = image_ndarray-np.min(image_ndarray)
    return ((temp_image)/((np.max(temp_image))*max_value)).astype(dtype)

def slice_image(image_ndarray: np.ndarray, indexes: list) -> np.ndarray:
    n_dim = len(image_ndarray.shape)
    slices = [slice(None)] * n_dim
    for i in range(len(indexes)):
        slices[i] = slice(indexes[i][0], indexes[i][1])
    slices = tuple(slices)
    sliced_image = image_ndarray[slices]
    return sliced_image


def mask_image_func(image_ndarray, mask_template_ndarray) -> np.ndarray:
    mask_ndarray = mask_template_ndarray
    return np.where(mask_ndarray == 255, image_ndarray, np.zeros(image_ndarray.shape))


def resize_image(patch_size, image):
    # only donwsampling, so use nearest neighbor that is faster to run
    resized_image = np.zeros(patch_size)
    for i in range(image.shape[0]):
        resized_image[i] = tf.image.resize(
            image[i], (patch_size[1], patch_size[2]
                       ), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
    # resized_image = tf.cast(resized_image, tf.float16)  # / 127.5 - 1.0
    return resized_image


class DataGen(keras.utils.Sequence):

    def __init__(self, image_list_csv, input_col, target_col,
                 batch_size,
                 num_batches=4,
                 patches_from_image=32,
                 patch_size=(16, 256, 256, 1),
                 min_precentage=0,
                 max_precentage=1,
                 image_number = None,
                 input_as_y=False,
                 output_as_x=False,
                 crop_edge=False,
                 mask=False,
                 mask_col='membrane_seg',
                 norm=True,
                 dilate=False,
                 dilate_kernel=np.ones((17, 17), np.uint8),
                 noise=False,
                 resize=True,
                 augment=False,
                 delete_cahce=False,
                 predictors=None,
                 pairs=False,
                 masking_pair = False,
                 cutoff = 0.0,
                 neg_ratio=1,
                 norm_type="std",
                 for_clf = False):

        self.new_path_origin = "/scratch/lionb@auth.ad.bgu.ac.il/{}/temp".format(
            os.environ.get('SLURM_JOB_ID'))
        if delete_cahce:
            shutil.rmtree(self.new_path_origin)

        self.df = DatasetMetadataSCV(image_list_csv, image_list_csv)
        self.n = self.df.get_shape()[0]
        self.input_col = input_col
        self.target_col = target_col
        self.batch_size = batch_size
        self.num_batches = num_batches
        # if 1 resize image to patch size if >1 then sample patches with patch size
        self.patches_from_image = patches_from_image
        self.patch_size = patch_size
        self.min_precentage = min_precentage
        self.max_precentage = max_precentage
        self.image_number = image_number
        self.input_as_y = input_as_y  # output of target and input in Y
        self.output_as_x = output_as_x  # input of target and input in X
        self.crop_edge = crop_edge
        self.mask = mask
        self.mask_col = mask_col
        self.norm = norm
        self.dilate = dilate
        self.dilate_kernel = dilate_kernel
        self.noise = noise
        self.resize = resize
        self.augment = augment
        self.predictors = predictors
        self.pairs = pairs
        self.row_ratio = 4
        self.col_ratio = 4
        self.neg_ratio = neg_ratio
        self.masking_pair = masking_pair
        self.cutoff = cutoff
        self.norm_type = norm_type
        self.for_clf = for_clf

        self.dists = {}
        self.patches_from_image = patches_from_image
        self.buffer_size = self.num_batches*self.batch_size*self.patches_from_image
        if self.pairs:
            self.buffer_size = self.buffer_size*self.row_ratio*self.col_ratio*(self.neg_ratio+1)
            self.input_buffer = np.zeros((self.buffer_size, self.patch_size[0],
                int(self.patch_size[1]/self.row_ratio), int(self.patch_size[2]/self.col_ratio), self.patch_size[-1]))
            self.target_buffer = np.zeros((self.buffer_size, self.patch_size[0],
                int(self.patch_size[1]/self.row_ratio), int(self.patch_size[2]/self.col_ratio), self.patch_size[-1]))
            # self.pred_buffer = np.zeros((self.buffer_size*self.row_ratio*self.col_ratio*(self.neg_ratio), self.patch_size[0],
            #     int(self.patch_size[1]/self.row_ratio), int(self.patch_size[2]/self.col_ratio), self.patch_size[-1]))
            self.labels_buffer = np.zeros(
                (self.buffer_size, 1))
        else:
            if (self.patch_size[0] == 1):  # 2D image
                two_d_patch_size = self.patch_size[1:]
                self.input_buffer = np.zeros(
                    (self.buffer_size, *two_d_patch_size))
                self.target_buffer = np.zeros(
                    (self.buffer_size, *two_d_patch_size))
            else:
                self.input_buffer = np.zeros(
                    (self.buffer_size, *self.patch_size))
                self.target_buffer = np.zeros(
                    (self.buffer_size, *self.patch_size))
                self.pred_buffer = np.zeros(
                    (self.buffer_size, *self.patch_size))
            if self.for_clf:
                self.labels_buffer = np.zeros((self.buffer_size, 1))

        if (not os.path.exists(self.new_path_origin)):
            os.makedirs(self.new_path_origin)

        self.fill_buffer()

    def get_image_from_ssd(self, file_path, prefix, k):
        file_name = file_path.split('/')[-1]
        new_file_path = "{}/{}_{}_{}".format(
            self.new_path_origin, prefix, k, file_name)
        if os.path.exists(new_file_path):
            image_ndarray = ImageUtils.image_to_ndarray(
                ImageUtils.imread(new_file_path))
            return image_ndarray, new_file_path
        return None, new_file_path

    def on_epoch_end(self):
        self.fill_buffer()

    def fill_samples(self, i):
        j = 0
        while j < self.batch_size:
            image_index = None
            try:
                if self.image_number is None:
                    image_index = np.random.randint(
                        int(self.n*self.min_precentage), int(self.n*self.max_precentage))
                    # print("adding image {} to buffer\n".format(image_index))
                else:
                    image_index = self.image_number

                image_path = self.df.get_item(image_index, 'path_tiff')

                if (self.augment):
                    k = np.random.random_integers(0, 3)
                else:
                    k = 0

                input_image, new_input_path = self.get_image_from_ssd(
                    image_path, self.input_col, k)
                if not self.for_clf:
                    target_image, new_target_path = self.get_image_from_ssd(
                        image_path, self.target_col, k)
                if self.predictors is not None:
                    pred_image, new_prediction_path = self.get_image_from_ssd(
                        image_path, "prediction", k)

                if (input_image is None or (target_image is None and not self.for_clf)):
                    image_ndarray = ImageUtils.image_to_ndarray(
                        ImageUtils.imread(image_path))
                    image_ndarray = image_ndarray.astype(np.float64)
                    if (self.augment):
                        image_ndarray = np.rot90(image_ndarray, axes=(2, 3), k=k)
                    if (self.crop_edge):
                        mask_index = int(self.df.get_item(
                            image_index, self.mask_col))
                        mask_template = ImageUtils.get_channel(
                            image_ndarray, mask_index)
                        image_ndarray = ImageUtils.crop_edges(
                            image_ndarray, mask_template)

                    mask_image = None
                    if (self.mask):
                        mask_index = int(self.df.get_item(
                            image_index, self.mask_col))
                        mask_image = ImageUtils.get_channel(
                            image_ndarray, mask_index)
                        if (self.dilate):
                            for h in range(mask_image.shape[0]):
                                mask_image[0, h, :, :] = cv2.dilate(mask_image[0, h, :, :].astype(
                                    np.uint8), self.dilate_kernel)

                    channel_index = int(self.df.get_item(
                        image_index, self.input_col))
                    input_image = ImageUtils.get_channel(
                        image_ndarray, channel_index)
                    if (self.mask):
                        input_image = mask_image_func(input_image, mask_image)
                    if (self.for_clf):
                        mask_index = int(self.df.get_item(image_index, 'structure_seg'))
                        masked_input_image = ImageUtils.get_channel(image_ndarray, mask_index)      
                    if (self.norm):
                        if self.norm_type == "std":
                            # input_image = ImageUtils.normalize(
                            #     input_image, max_value=1.0, dtype=np.float16)
                            mean = np.mean(input_image,dtype=np.float64)
                            std = np.std(input_image,dtype=np.float64)
                            if (np.isnan(mean) or np.isnan(std) or np.isinf(mean) or np.isinf(std)):
                                # raise Exception("Error calculating mean or std")
                                # print("Calculating mean and std again, input_image:{}".format(image_path))  
                                max_var = np.max(input_image!=np.inf)
                                input_image = np.where(input_image==np.inf,max_var,input_image)
                                mean = np.mean(input_image,dtype=np.float64)
                                std = np.std(input_image,dtype=np.float64)
                            self.dists[new_input_path] = {"mean":mean,"std":std}
                            # print("{} mean:{}, std:{}".format(new_input_path,mean,std))
                            input_image = (input_image-mean)/std
                        else:
                            input_image = normalize(input_image, max_value=1.0, dtype=np.float32)

                    if (self.target_col == self.input_col or self.for_clf):
                        target_image = input_image
                    else:
                        channel_index = int(self.df.get_item(
                            image_index, self.target_col))
                        target_image = ImageUtils.get_channel(
                            image_ndarray, channel_index)
                        if (self.mask):
                            target_image = mask_image_func(
                                target_image, mask_image)
                        if (self.dilate):
                            for h in range(target_image.shape[1]):
                                target_image[0, h, :, :] = cv2.dilate(target_image[0, h, :, :].astype(
                                    np.uint8), self.dilate_kernel)                            
                        else:
                            if (self.norm):
                                if self.norm_type == "std":
                                    # target_image = ImageUtils.normalize(
                                    #     target_image, max_value=1.0, dtype=np.float16)
                                    mean = np.mean(target_image,dtype=np.float64)
                                    std = np.std(target_image,dtype=np.float64)
                                    if (np.isnan(mean) or np.isnan(std) or np.isinf(mean) or np.isinf(std)):
                                        # raise Exception("Error calculating mean or std")  
                                        # print("Calculating mean and std again, target_image:{}".format(image_path))   
                                        max_var = np.max(target_image!=np.inf)
                                        target_image = np.where(target_image==np.inf,max_var,target_image)
                                        mean = np.mean(target_image,dtype=np.float64)
                                        std = np.std(target_image,dtype=np.float64)                        
                                    self.dists[new_target_path] = {"mean":mean,"std":std}
                                    # print("{} mean:{}, std:{}".format(new_target_path,mean,std))
                                    target_image = (target_image-mean)/std
                                else:
                                    target_image = normalize(target_image, max_value=1.0, dtype=np.float32)
                    if (self.noise):
                        target_image += np.random.normal(0,
                                                        0.05, size=target_image.shape)
                        target_image = np.clip(target_image, 0, 1)
                    input_image = np.expand_dims(input_image[0], axis=-1)
                    target_image = np.expand_dims(target_image[0], axis=-1)
                    ImageUtils.imsave(input_image, new_input_path)
                    if not self.for_clf:
                        ImageUtils.imsave(target_image, new_target_path)

                if self.pairs:
                    input_image = ImageUtils.to_shape(input_image, self.patch_size)
                    target_image = ImageUtils.to_shape(target_image, self.patch_size)
                    input_patches = patchify(input_image, (self.patch_size[0], int(self.patch_size[1]/self.row_ratio), int(self.patch_size[2]/self.col_ratio),self.patch_size[-1]), step=int(self.patch_size[2]/self.col_ratio))
                    target_patches = patchify(target_image, (self.patch_size[0], int(self.patch_size[1]/self.row_ratio), int(self.patch_size[2]/self.col_ratio),self.patch_size[-1]), step=int(self.patch_size[2]/self.col_ratio))
                    ### add masking
                    mask_vol = 4
                    mask_ratio = 0.75 #0.75
                    num_masks = int((np.prod(input_patches.shape[3:-1])*mask_ratio)/mask_vol)
                    masking_ind_z = np.random.random_integers(0,int(self.patch_size[0]/mask_vol)-1,num_masks)
                    masking_ind_x = np.random.random_integers(0,int(self.patch_size[1]/mask_vol)-1,num_masks)
                    masking_ind_y = np.random.random_integers(0,int(self.patch_size[2]/mask_vol)-1,num_masks)
                    if self.masking_pair:
                        masked_target_patchs = np.copy(target_patches)
                        masked_input_patchs = np.copy(input_patches)
                        for w in range(num_masks):
                            # masked_target_patchs[:,:,:,:,masking_ind_z[w]*mask_vol:(masking_ind_z[w]+1)*mask_vol,masking_ind_x[w]*mask_vol:(masking_ind_x[w]+1)*mask_vol,masking_ind_y[w]*mask_vol:(masking_ind_y[w]+1)*mask_vol,0] = 0
                            masked_input_patchs[:,:,:,:,masking_ind_z[w]*mask_vol:(masking_ind_z[w]+1)*mask_vol,masking_ind_x[w]*mask_vol:(masking_ind_x[w]+1)*mask_vol,masking_ind_y[w]*mask_vol:(masking_ind_y[w]+1)*mask_vol,0] = 0
                    else:
                        masked_target_patchs = target_patches
                        masked_input_patchs = input_patches
                    for l in range(input_patches.shape[1]):
                        for m in range(input_patches.shape[2]):
                            pos = i*(self.batch_size*input_patches.shape[1]*input_patches.shape[2]*(self.neg_ratio+1))+j*(input_patches.shape[1]*input_patches.shape[2]*(self.neg_ratio+1))+l*(input_patches.shape[2]*(self.neg_ratio+1))+m*(self.neg_ratio+1)
                            self.input_buffer[pos] = masked_input_patchs[0,l,m]
                            self.target_buffer[pos] = masked_target_patchs[0,l,m]
                            self.labels_buffer[pos] = 1
                            for nr in range(1,self.neg_ratio+1):
                                l_random = np.random.random_integers(0,input_patches.shape[1]-1)
                                m_random = np.random.random_integers(0,input_patches.shape[2]-1)
                                neg_label = 0
                                if (np.max(target_patches[0,l_random,m_random])<=self.cutoff and  np.max(target_patches[0,l,m])<=self.cutoff):
                                    neg_label = 1
                                self.input_buffer[pos+nr] = masked_input_patchs[0,l,m]
                                self.target_buffer[pos+nr] = masked_target_patchs[0,l_random,m_random]
                                self.labels_buffer[pos+nr] = neg_label
                                
                else:
                    if (self.patches_from_image > 1):
                        # Sample patches
                        # input_image = ImageUtils.to_shape(input_image,input_image.shape,None, min_shape=self.patch_size)
                        # target_image = ImageUtils.to_shape(target_image,input_image.shape,None, min_shape=self.patch_size)
                        for k in range(self.patches_from_image):
                            patch_indexes = []
                            for o in range(len(self.patch_size)-1):
                                random_patch_index = np.random.randint(
                                    input_image.shape[o]-self.patch_size[o]+1)
                                patch_indexes.append(
                                    (random_patch_index, random_patch_index+self.patch_size[o]))
                            input_patch = slice_image(
                                input_image, [*patch_indexes, (None, None)])
                            target_patch = slice_image(
                                target_image, [*patch_indexes, (None, None)])
                            if (self.patch_size[0] == 1):  # 2D image
                                input_patch = input_patch[0]
                                target_patch = target_patch[0]
                            pos = i*(self.batch_size*self.patches_from_image)+j*(self.patches_from_image)+k
                            self.input_buffer[pos] = input_patch
                            self.target_buffer[pos] = target_patch
                            if self.for_clf:
                                masked_input_patch = slice_image(masked_input_image, [*patch_indexes, (None, None)])                                
                                label = 12
                                if np.sum(masked_input_patch/255.) > 1:
                                    label = self.df.get_item(image_index, 'label')
                                self.labels_buffer[pos] = label
                    else:
                        if (self.resize):
                            # Resize to patch size
                            # input_image = ImageUtils.to_shape(
                            #     input_image, (16, *self.patch_size[1:]))
                            # target_image = ImageUtils.to_shape(
                            #     target_image, (16, *self.patch_size[1:]))
                            input_image = ImageUtils.to_shape(
                                input_image, (16,256,256,1))
                            target_image = ImageUtils.to_shape(
                                target_image, (16,256,256,1))
                            # ndimage.zoom(input_image,[self.patch_size[0]/input_image.shape[0],1,1,1],mode="constant",cval=0,order=1,prefilter=False)
                            input_image = resize_image(
                                self.patch_size, input_image)
                            # ndimage.zoom(target_image,[self.patch_size[0]/target_image.shape[0],1,1,1],mode="constant",cval=0,order=1,prefilter=False)
                            target_image = resize_image(
                                self.patch_size, target_image)
                        else:
                            input_image = ImageUtils.to_shape(
                                input_image, self.patch_size)
                            target_image = ImageUtils.to_shape(
                                target_image, self.patch_size)
                            if self.predictors is not None:
                                if pred_image is None and self.predictors is not None:
                                    if isinstance(self.predictors,dict):
                                        organelle = image_path.split('/')[-1]
                                        organelle = organelle.split("_")[0]
                                        pred_image = self.predictors[organelle](np.expand_dims(input_image, axis=0)).numpy()

                                    else:
                                        channel_index = int(self.df.get_item(image_index, "channel_dna"))
                                        pred_image = ImageUtils.get_channel(image_ndarray, channel_index)
                                    if (self.norm):
                                        if self.norm_type == "std":
                                            mean = np.mean(pred_image,dtype=np.float64)
                                            std = np.std(pred_image,dtype=np.float64)
                                            if (np.isnan(mean) or np.isnan(std) or np.isinf(mean) or np.isinf(std)):
                                                # raise Exception("Error calculating mean or std")  
                                                # print("Calculating mean and std again, target_image:{}".format(image_path))   
                                                max_var = np.max(pred_image!=np.inf)
                                                pred_image = np.where(pred_image==np.inf,max_var,pred_image)
                                                mean = np.mean(pred_image,dtype=np.float64)
                                                std = np.std(pred_image,dtype=np.float64)                        
                                            pred_image = (pred_image-mean)/std
                                            pred_image = np.expand_dims(pred_image[0], axis=-1)
                                        else:
                                            pred_image = normalize(pred_image, max_value=1.0, dtype=np.float32)                                    
                                    ImageUtils.imsave(pred_image, new_prediction_path)
                                    self.pred_buffer[i*self.batch_size+j *self.patches_from_image] = pred_image

                        self.input_buffer[i*self.batch_size+j *
                                        self.patches_from_image] = input_image
                        self.target_buffer[i*self.batch_size+j *
                                        self.patches_from_image] = target_image
                        if self.for_clf:
                            label = self.df.get_item(image_index, 'label')
                            self.labels_buffer[i*self.batch_size+j * self.patches_from_image] = label
                j+=1
            except:
                print("could not load image {} , trying another one.\n".format(image_index))

            # if (old_image_path != image_path):
            #     os.remove(image_path)
          

    def fill_buffer(self):
        threads = []
        num_threads = 4
        for i in tqdm(range(self.num_batches)):
            if (num_threads > 1):
                thread = threading.Thread(target=self.fill_samples, args=[i])
                thread.start()
                threads.append(thread)

                if ((i+1) % num_threads == 0):
                    for t in (range(len(threads))):
                        threads[t].join()
                    threads = []
            else:
                self.fill_samples(i)
        # print(self.dists)

        for thread in threads:
            thread.join()
        if  self.pairs:
            self.input_buffer, self.target_buffer, self.labels_buffer = sklearn.utils.shuffle(
            self.input_buffer, self.target_buffer, self.labels_buffer)
        else:
            self.input_buffer, self.target_buffer = sklearn.utils.shuffle(
            self.input_buffer, self.target_buffer)

    def __getitem__(self, index):
        num_samples = int(self.buffer_size/(self.num_batches))
        if (self.patches_from_image > 1):
            num_samples = int(self.buffer_size/(self.num_batches*self.patches_from_image))
        if self.pairs:
            num_samples = int(self.buffer_size/(self.num_batches*self.col_ratio))
        X = self.input_buffer[index*num_samples:(index+1)*num_samples]
        # y1 = deepcopy(self.input_buffer[index*self.batch_size:(index+1)*self.batch_size])
        
        Y = self.target_buffer[index*num_samples:(index+1)*num_samples]
        if (self.input_as_y):
            Y = [X, Y]
        if (self.output_as_x):
            X = [X, Y]
        if self.predictors is not None:
            P = self.pred_buffer[index*num_samples:(index+1)*num_samples]
            X = [X, P]
        if self.pairs:
            L = self.labels_buffer[index*num_samples:(index+1)*num_samples]
            X = [X,Y]
            Y = L
        if self.for_clf:
            X = X
            Y = self.labels_buffer[index*num_samples:(index+1)*num_samples]
        return X, Y

    def __len__(self):
        if (self.patches_from_image > 1):
            return self.num_batches*self.patches_from_image
        if self.pairs:
            return self.num_batches*self.col_ratio
        return self.num_batches
