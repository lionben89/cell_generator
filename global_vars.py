model_type = "AAE"
model_path = "./model_aae_dim_16_256_256_ld_64_loss_ce_config_ne2ne"
patch_size = (16,256,256,1) ## 2D: (1,*,*,1) , 3D: (*,*,*,1)
latent_dim = 64
number_epochs = 1000
batch_norm = False

input = "channel_target"
target = "channel_target"
# input = "channel_signal"
# target = "channel_signal"

# train_ds_path = "/sise/home/lionb/cell_generator/not_masked/image_list_train.csv"
train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nuclear-envelope/image_list_train.csv"
# test_ds_path = "/sise/home/lionb/cell_generator/not_masked/image_list_test.csv"
test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nuclear-envelope/image_list_test.csv"