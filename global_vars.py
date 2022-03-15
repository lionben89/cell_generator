model_type = "UNET"
model_path = "./aae_model_ne_15_03_22" #mask bn, ne, bf_2_ne, _ld_256
latent_to_latent_model_path = "./model_ltol_dim_16_256_256_ld_64_config_bf2ne"
unet_model_path = "./unet_model_ne_15_03_22" #no mask with bn
sg_model_path = "./sg_model_ne_08_03_22" #no mask with bn, _ut_adaptor
stylegan_path = "./stylegan"
patch_size = (16,256,256,1) ## 2D: (1,*,*,1) , 3D: (*,*,*,1)
latent_dim = 64
number_epochs = 2000
batch_size = 8
batch_norm = True

# input = "channel_target"
input = "channel_signal"
target = "channel_target"
# target = "channel_signal"
#Nucleolus-(Granular-Component)
# train_ds_path = "/sise/home/lionb/cell_generator/not_masked/image_list_train.csv"
train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nuclear-envelope/image_list_train.csv" ##_fl loss mae noise=False no from_logits
# test_ds_path = "/sise/home/lionb/cell_generator/not_masked/image_list_test.csv"
test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nuclear-envelope/image_list_test.csv" ##_fl loss mae noise=False no from_logits