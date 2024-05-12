import numpy as np

# function for core algorithm
from aicssegmentation.core.vessel import filament_3d_wrapper, filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, edge_preserving_smoothing_3d, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects,binary_closing, ball , dilation   # function for post-processing (size filter)

from aicssegmentation.core.seg_dot import dot_3d_wrapper, dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d, image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.utils import topology_preserving_thinning, hole_filling
from aicssegmentation.core.MO_threshold import MO
from skimage.segmentation import watershed

class SegProto:
    def segment(image_ndarray):
        pass   
    
class SegMicrotubules(SegProto):
    def __init__(self):
        super().__init__()
        
    def segment(self,image_ndarray):
        intensity_scaling_param = [1.5,8]
        seg_image_ndarray = intensity_normalization(image_ndarray, scaling_param=intensity_scaling_param)
        seg_image_ndarray = edge_preserving_smoothing_3d(seg_image_ndarray)
        
        f3_param = [[1, 0.01]]
        seg_image_ndarray = filament_3d_wrapper(seg_image_ndarray, f3_param)
        
        minArea = 4
        seg_image_ndarray = remove_small_objects(seg_image_ndarray>0, min_size=minArea, connectivity=1)
        
        seg_image_ndarray = seg_image_ndarray >0
        seg_image_ndarray=seg_image_ndarray.astype(np.uint8)
        seg_image_ndarray[seg_image_ndarray>0]=255
        return seg_image_ndarray

class SegMyosin(SegProto):
    def __init__(self):
        super().__init__()
        
    def segment(self,image_ndarray):
        intensity_scaling_param = [2.5,17]
        seg_image_ndarray = intensity_normalization(image_ndarray, scaling_param=intensity_scaling_param)
        seg_image_ndarray = edge_preserving_smoothing_3d(seg_image_ndarray)
        
        f3_param = [[2,0.2],[1,0.015]]
        seg_image_ndarray = filament_3d_wrapper(seg_image_ndarray, f3_param)
        
        minArea = 4
        seg_image_ndarray = remove_small_objects(seg_image_ndarray>0, min_size=minArea, connectivity=1)
        
        seg_image_ndarray = seg_image_ndarray >0
        seg_image_ndarray=seg_image_ndarray.astype(np.uint8)
        seg_image_ndarray[seg_image_ndarray>0]=255
        return seg_image_ndarray    
    
class SegActinFilaments(SegProto):
    def __init__(self):
        super().__init__()
        
    def segment(self,image_ndarray):
        intensity_scaling_param = [3, 15]
        seg_image_ndarray = intensity_normalization(image_ndarray, scaling_param=intensity_scaling_param)
        seg_image_ndarray = edge_preserving_smoothing_3d(seg_image_ndarray)
        
        f3_param = [[2,0.1],[1,0.04]]
        seg_image_ndarray = filament_3d_wrapper(seg_image_ndarray, f3_param)
        
        minArea = 4
        seg_image_ndarray = remove_small_objects(seg_image_ndarray>0, min_size=minArea, connectivity=1)
        
        seg_image_ndarray = seg_image_ndarray >0
        seg_image_ndarray=seg_image_ndarray.astype(np.uint8)
        seg_image_ndarray[seg_image_ndarray>0]=255
        return seg_image_ndarray   
    
class SegTightJunctions(SegProto):
    def __init__(self):
        super().__init__()
        
    def segment(self,image_ndarray):
        intensity_scaling_param = [3, 17]
        seg_image_ndarray = intensity_normalization(image_ndarray, scaling_param=intensity_scaling_param)
        
        gaussian_smoothing_sigma = 1
        seg_image_ndarray = image_smoothing_gaussian_3d(seg_image_ndarray, sigma=gaussian_smoothing_sigma)
        
        f3_param = [[1.5, 0.2]]
        seg_image_ndarray = filament_3d_wrapper(seg_image_ndarray, f3_param)
        
        minArea = 4
        seg_image_ndarray = remove_small_objects(seg_image_ndarray>0, min_size=minArea, connectivity=1)
        
        seg_image_ndarray = seg_image_ndarray >0
        seg_image_ndarray=seg_image_ndarray.astype(np.uint8)
        seg_image_ndarray[seg_image_ndarray>0]=255
        return seg_image_ndarray    
    
class SegGolgi(SegProto):
    def __init__(self):
        super().__init__()
        
    def segment(self,image_ndarray):
        intensity_scaling_param = [9,19]
        gaussian_smoothing_sigma = 1
        seg_image_ndarray = intensity_normalization(image_ndarray, scaling_param=intensity_scaling_param)
        seg_image_ndarray = image_smoothing_gaussian_3d(seg_image_ndarray, sigma=gaussian_smoothing_sigma)
        
        seg_image_ndarray1, object_for_debug = MO(seg_image_ndarray, global_thresh_method='tri', object_minArea=1200, return_object=True)

        thin_dist_preserve=1.6
        thin_dist=1
        seg_image_ndarray1 = topology_preserving_thinning(seg_image_ndarray1>0, thin_dist_preserve, thin_dist)    
        
        s3_param = [[1.6, 0.02]]
        seg_image_ndarray2 = dot_3d_wrapper(seg_image_ndarray, s3_param) 
        
        seg_image_ndarray = np.logical_or(seg_image_ndarray2>0, seg_image_ndarray1)           
        
        minArea = 10
        seg_image_ndarray = remove_small_objects(seg_image_ndarray>0, min_size=minArea, connectivity=1)
        
        seg_image_ndarray = seg_image_ndarray >0
        seg_image_ndarray=seg_image_ndarray.astype(np.uint8)
        seg_image_ndarray[seg_image_ndarray>0]=255
        return seg_image_ndarray  
    
class SegER(SegProto):
    def __init__(self):
        super().__init__()
        
    def segment(self,image_ndarray):
        intensity_scaling_param = [2.5, 7.5]
        gaussian_smoothing_sigma = 1
        seg_image_ndarray = intensity_normalization(image_ndarray, scaling_param=intensity_scaling_param)
        
        seg_image_ndarray = edge_preserving_smoothing_3d(seg_image_ndarray)
        
        f2_param = [[1, 0.15]]
        seg_image_ndarray = filament_2d_wrapper(seg_image_ndarray, f2_param)
        
        minArea = 20
        seg_image_ndarray = remove_small_objects(seg_image_ndarray>0, min_size=minArea, connectivity=1)
        
        seg_image_ndarray = seg_image_ndarray >0
        seg_image_ndarray=seg_image_ndarray.astype(np.uint8)
        seg_image_ndarray[seg_image_ndarray>0]=255
        return seg_image_ndarray    
    
class SegLysosome(SegProto):
    def __init__(self):
        super().__init__()
        
    def segment(self,image_ndarray):
        intensity_scaling_param = [3, 19]
        gaussian_smoothing_sigma = 1
        seg_image_ndarray = intensity_normalization(image_ndarray, scaling_param=intensity_scaling_param)
        seg_image_ndarray = image_smoothing_gaussian_slice_by_slice(seg_image_ndarray, sigma=gaussian_smoothing_sigma)
        
        s2_param = [[5,0.09], [2.5,0.07], [1,0.01]]
        seg_image_ndarray1 = dot_2d_slice_by_slice_wrapper(seg_image_ndarray, s2_param)
        
        f2_param = [[1, 0.15]]
        seg_image_ndarray2 = filament_2d_wrapper(seg_image_ndarray, f2_param)
        
        seg_image_ndarray = np.logical_or(seg_image_ndarray1, seg_image_ndarray2)
        
        fill_2d = True
        fill_max_size = 1600
        seg_image_ndarray = hole_filling(seg_image_ndarray, 0, fill_max_size, fill_2d)
        
        minArea = 215
        seg_image_ndarray = remove_small_objects(seg_image_ndarray>0, min_size=minArea, connectivity=1)
        
        seg_image_ndarray = seg_image_ndarray >0
        seg_image_ndarray=seg_image_ndarray.astype(np.uint8)
        seg_image_ndarray[seg_image_ndarray>0]=255
        return seg_image_ndarray              