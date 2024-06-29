import os
import threading
import numpy as np
from cell_imaging_utils.datasets_metadata.table.datasetes_metadata_csv import DatasetMetadataSCV
from cell_imaging_utils.image.image_utils import ImageUtils
import quilt3
import shutil
from collections import OrderedDict

"""This script will download the field of views (FOVs) data from the Allen Inst. S3 and will stack all the relevant channels for that FOV"""
## global vars
#number of threads to run
num_threads=4
# S3 bucket
data_provider = quilt3.Bucket("s3://allencell")
#Path to data location in bucket
download_path = "aics/pipeline_integrated_cell/"
#where to save downloaded data
storage_root = "/sise/assafzar-group/assafzar/full_cells_fovs/"
#temp location to save data that is being processed (SSD memory)
temp_storage_root = "/scratch/lionb@auth.ad.bgu.ac.il/{}/full_cells_fovs/".format(os.environ.get('SLURM_JOB_ID')) ##"/storage/users/assafzar/single_cells_fovs/"
#path to metadata.csv
datasets_metadata_dir = "{}metadata.csv".format(storage_root)
#max number of images to download
num_of_images_per_organelle = 200
resacle_z = 3
#what organelles to download
organelles={"Endosomes":[]}#{"Desmosomes":[],"Golgi":[],"Microtubules":[],"Nuclear-envelope":[],"Actin-filaments":[],"Plasma-membrane":[],"Nucleolus-(Dense-Fibrillar-Component)":[],"Mitochondria":[],"Endoplasmic-reticulum":[],"Tight-junctions":[],"Nucleolus-(Granular-Component)":[],"Actomyosin-bundles":[]}

## instructions - channels in outputs images
# 0-BF roi
# 1 dna raw roi
# 2 membrane raw roi
# 3 structure raw roi
# 4 dna seg roi
# 5 membrane seg roi
# 6 structure seg roi
##

## roi is [z_min,z_max,y_min,y_max,x_min,x_max]
## fov to crop is 1:3 ratio 
## metadata.data.loc[metadata.data['structure_name'] == 'GJA1']
## not found 'ATP2A2','HIST1H2BJ','NUP153','SMC1A','SON'

#just create csvs without images
only_csvs = True
#dowlowd and processed images that been processed in the past
override = False


class LRUCache:
 
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
 
    # we return the value of the key
    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]
 
    # first, we add / update the key by conventional methods.
    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)


ongoing_downloads = {}
ongoing_downloads_lock = threading.Lock()

cache = LRUCache(4)
cache_lock = threading.Lock()

proteain_to_orgnelle_dict = {'Plasma-membrane':'AAVS1', 'Actin-filaments':'ACTN1', 'Adherens-junctions':'CTNNB1', 'Desmosomes':'DSP',
       'Nucleolus-(Dense-Fibrillar-Component)':'FBL', 'Gap-junctions':'GJA1', 'Lysosome':'LAMP1', 'Nuclear-envelope':'LMNB1', 'Actomyosin-bundles':'MYH10', 'Nucleolus-(Granular-Component)':'NPM1',
        'Matrix-adhesions':'PXN', 'Endosomes':'RAB5A', 'Endoplasmic-reticulum':'SEC61B', 'Peroxisomes':'SLC25A17',
       'Golgi':'ST6GAL1', 'Tight-junctions':'TJP1', 'Mitochondria':'TOMM20', 'Microtubules':'TUBA1B'}

def download_metadata_if_not_exist():
    try:
        if not os.path.exists(datasets_metadata_dir):
            data_provider.fetch("{}metadata.csv".format(download_path), datasets_metadata_dir)
    except:
        print("Downloading metadata file from {} failed".format("{}/metadata.csv".format(download_path)))

def download_image_if_not_exists(download_path,storage_path):
    global ongoing_downloads_lock,ongoing_downloads
    ongoing_downloads_lock.acquire()
    if (not os.path.exists(storage_path) and not(storage_path in ongoing_downloads)):
        ongoing_downloads[storage_path] = threading.Lock()
        ongoing_downloads[storage_path].acquire()
        ongoing_downloads_lock.release()
        data_provider.fetch(download_path,storage_path)
        ongoing_downloads[storage_path].release()
    else:
        if (not(storage_path in ongoing_downloads)):
            ongoing_downloads[storage_path] = threading.Lock()
        ongoing_downloads_lock.release()
        ongoing_downloads[storage_path].acquire()
        ongoing_downloads[storage_path].release()

def get_image(storage_temp_fov):
  cache_lock.acquire()
  image = cache.get(storage_temp_fov)
  if (not isinstance(image,np.ndarray)):
    print("loaded to cache".format(storage_temp_fov))
    image = ImageUtils.image_to_ndarray(ImageUtils.imread(storage_temp_fov))
    cache.put(storage_temp_fov,image)
  cache_lock.release()
  return image

def download_and_create_image(fov_path,fov_channel,structure_fl_channel,dna_fl_channel,mem_fl_channel,structure_seg_path,dna_seg_path,mem_seg_path,new_image_path):
## instructions - channels in outputs images
# 0-FOV roi
# 1 dna raw roi
# 2 membrane raw roi
# 3 structure raw roi
# 4 dna seg roi
# 5 membrane seg roi
# 6 structure seg roi
##
    if (not only_csvs):
        if ((not override) and os.path.exists(new_image_path)):
            print("image already exists: {}".format(new_image_path))
            return
            
        print("Creating image {}".format(new_image_path))
        new_image = None
        
        storage_temp_fov = "{}temp/{}".format(temp_storage_root,fov_path)
        download_image_if_not_exists("{}{}".format(download_path,fov_path),storage_temp_fov)      
        fov_ndarray = get_image(storage_temp_fov)
        sliced_fov_ndarray = ImageUtils.get_channel(fov_ndarray,int(fov_channel))
        new_image = sliced_fov_ndarray     
        
        dna_channel_ndarray = ImageUtils.get_channel(fov_ndarray,int(dna_fl_channel))
        new_image = ImageUtils.add_channel(new_image,dna_channel_ndarray)
        
        mem_channel_ndarray = ImageUtils.get_channel(fov_ndarray,int(mem_fl_channel))
        new_image = ImageUtils.add_channel(new_image,mem_channel_ndarray)
        
        structure_channel_ndarray = ImageUtils.get_channel(fov_ndarray,int(structure_fl_channel))
        new_image = ImageUtils.add_channel(new_image,structure_channel_ndarray)
        
        storage_temp_seg = "{}temp/{}".format(temp_storage_root,dna_seg_path)
        download_image_if_not_exists("{}{}".format(download_path,dna_seg_path),storage_temp_seg)      
        dna_seg_ndarray = get_image(storage_temp_seg)
        new_image = ImageUtils.add_channel(new_image,dna_seg_ndarray)
        
        storage_temp_seg = "{}temp/{}".format(temp_storage_root,mem_seg_path)
        download_image_if_not_exists("{}{}".format(download_path,mem_seg_path),storage_temp_seg)      
        mem_seg_ndarray = get_image(storage_temp_seg)
        new_image = ImageUtils.add_channel(new_image,mem_seg_ndarray)
        
        storage_temp_seg = "{}temp/{}".format(temp_storage_root,structure_seg_path)
        download_image_if_not_exists("{}{}".format(download_path,structure_seg_path),storage_temp_seg)      
        structure_seg_ndarray = get_image(storage_temp_seg)
        new_image = ImageUtils.add_channel(new_image,structure_seg_ndarray)
        
         
        print("saving image: {}".format(new_image_path))
        ImageUtils.imsave(new_image.astype(np.float16),new_image_path)
        print("saved image: {}".format(new_image_path))

def delete_temp_folder():
  if (os.path.exists(temp_storage_root)):
    print("Deleting temp folder {}temp".format(temp_storage_root))
    with os.scandir("{}temp".format(temp_storage_root)) as entries:
      for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
          shutil.rmtree(entry.path)
        else:
          os.remove(entry.path)
        print("Finish deleting temp folder {}temp".format(temp_storage_root))

def cretae_metadata_file(metadata,organelle):
    organelle_dir = "{}{}".format(storage_root,organelle)
    organelle_metadata_path = "{}/{}.csv".format(organelle_dir,organelle)
    print("Creating {}".format(organelle_metadata_path))
    if (not os.path.exists(organelle_dir)):
        os.makedirs(organelle_dir)
    if (os.path.exists(organelle_metadata_path)):
        os.remove(organelle_metadata_path)
        
    organelle_metadata = DatasetMetadataSCV(organelle_metadata_path)
    proteain = proteain_to_orgnelle_dict[organelle]
    
    sliced_metadata = metadata.data.loc[(metadata.data['Gene'] == proteain)].drop_duplicates(subset=['WellId']).reset_index(drop=True)
    num_images = min(num_of_images_per_organelle,sliced_metadata.shape[0])
    sliced_metadata = sliced_metadata[:num_images]
    # sliced_metadata = sliced_metadata.drop(columns=sliced_metadata.columns[46:]) #data after 47 column is not interesting
    # sliced_metadata = sliced_metadata.reset_index()
    organelle_metadata.set_data(sliced_metadata)
    return organelle_metadata

def download_and_create_dataset():
    download_metadata_if_not_exist()
    print("Opening metadata file {}".format(datasets_metadata_dir))
    metadata  = DatasetMetadataSCV(datasets_metadata_dir,datasets_metadata_dir)
    print("Finish Opening metadata file {}...".format(datasets_metadata_dir))
    for organelle in organelles.keys():
        organelle_metadata = cretae_metadata_file(metadata,organelle)
        count = 0
        threads = []
        organelle_metadata.data['combined_image_storage_path']=None
        for i in range(organelle_metadata.get_shape()[0]):
            fov_path=organelle_metadata.get_item(i,'SourceReadPath')
            fov_channel = organelle_metadata.get_item(i,'ChannelNumberBrightfield')
            structure_fl_channel = organelle_metadata.get_item(i,'ChannelNumberStruct') 
            dna_fl_channel = organelle_metadata.get_item(i,'ChannelNumber405') 
            mem_fl_channel = organelle_metadata.get_item(i,'ChannelNumber638')
            structure_seg_path = organelle_metadata.get_item(i,'StructureSegmentationReadPath') 
            dna_seg_path = organelle_metadata.get_item(i,'NucleusSegmentationReadPath') 
            mem_seg_path = organelle_metadata.get_item(i,'MembraneSegmentationReadPath') 
            new_image_path = "{}{}/{}_{}.tiff".format(storage_root,organelle,organelle,i)
            thread=threading.Thread(target=download_and_create_image,args=(fov_path,fov_channel,structure_fl_channel,dna_fl_channel,mem_fl_channel,structure_seg_path,dna_seg_path,mem_seg_path,new_image_path))
            organelle_metadata.set_item(i,'combined_image_storage_path',new_image_path)
            thread.start()
            threads.append(thread)
            count+=1;
            if ((i+1)%num_threads == 0):
              for thread in threads:
                thread.join()
                threads = []
                
            if ((i+1)%(num_threads*8) == 0):
              delete_temp_folder()
        
        for thread in threads:
            thread.join()
                
        organelle_metadata.create()
        print("Finish creating metadata csv: {}".format(organelle))   
        delete_temp_folder()             
                
                
def run():
    download_and_create_dataset()
    
run()