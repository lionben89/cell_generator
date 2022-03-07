model_type = "SG"
model_path = "./model_aae_ne" ##_noise, _Z, _fl
latent_to_latent_model_path = "./model_ltol_dim_16_256_256_ld_64_config_bf2ne"
unet_model_path = "./unet_model_nuclear_envelope_01_03_22" 
sg_model_path = "./sg_model_nuclear_envelope_02_03_22_with_sampling_unet_adaptor_aae_decoder_lp_freeze" #_adaptor_64_filters #_adaptor #_unet_adaptor_aae_decoder_lp_freeze_corriculum #unet_adaptor_aae_decoder_lp_freeze
stylegan_path = "./stylegan"
patch_size = (16,256,256,1) ## 2D: (1,*,*,1) , 3D: (*,*,*,1)
latent_dim = 64
number_epochs = 2000
batch_size = 4
batch_norm = False

# input = "channel_target"
input = "channel_signal"
target = "channel_target"
# target = "channel_signal"
#Nucleolus-(Granular-Component)
# train_ds_path = "/sise/home/lionb/cell_generator/not_masked/image_list_train.csv"
train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nuclear-envelope/image_list_train.csv" ##_fl loss mae noise=False no from_logits
# test_ds_path = "/sise/home/lionb/cell_generator/not_masked/image_list_test.csv"
test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nuclear-envelope/image_list_test.csv" ##_fl loss mae noise=False no from_logits