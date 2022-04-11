model_type = "UNET"
model_path = "./aae_model_ne_27_03_22_128_fl_next" #"./aae_model_ne_27_03_22_128_fl" #"./aae_model_ne_27_03_22_128_fl_next" #"./aae_model_ne_26_03_22_128" #"./aae_model_ne_27_03_22_128_next" #"./aae_model_ne_26_03_22_bf_1024"        #"./aae_model_ne_22_03_22_seg" #"./aae_model_ne_22_03_22_bf"
latent_to_latent_model_path = "./model_ltol_dim_16_256_256_ld_64_config_bf2ne"
unet_model_path = "./unet_model_x_bce_2_3_nobn_ne_seg"#_mse_2_3_nobn_ne_seg (loss=0.001 lr=0.00001) _mse_2_3_nobn(loss=0.01 lr=0.0001) _mse_2_3_nobn_ne(loss=0.01 lr=0.0001) _mae"./unet_model_06_04_22" #"./unet_model_ne_05_04_22_seg_bce"#"./unet_model_ne_05_04_22_seg_bce" #"./unet_model_ne_04_04_22_seg"
sg_model_path = "./sg_model_ne_02_04_22"
stylegan_path = "./stylegan"
eam_model_path = "./eam_model_ne_09_04_22"
patch_size = (16,256,256,1) ## 2D: (1,*,*,1) , 3D: (*,*,*,1)
latent_dim = 128 #128 #64 #1024
number_epochs = 10000
batch_size = 4
batch_norm = False

# input = "channel_target"
input = "channel_signal"
# input = "structure_seg"
# target = "channel_target"
# target = "channel_signal"
target = "structure_seg"
#Nucleolus-(Granular-Component)
# train_ds_path = "/sise/home/lionb/cell_generator/image_list_train.csv"
# train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nucleolus-(Granular-Component)/image_list_train.csv" #norm input
train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nuclear-envelope/image_list_train.csv" #no norm input
# train_ds_path = "/sise/assafzar-group/assafzar/training_from_segmentation/Nuclear-envelope/image_list_train.csv"
# test_ds_path = "/sise/home/lionb/cell_generator/image_list_test.csv"
# test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nucleolus-(Granular-Component)/image_list_test.csv"
test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nuclear-envelope/image_list_test.csv"
# test_ds_path = "/sise/assafzar-group/assafzar/training_from_segmentation/Nuclear-envelope/image_list_test.csv"