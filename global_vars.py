model_type = "AAE"
model_path = "./aae_single_cell_2d_not_masked"
patch_size = (32,64,64,1) ## 2D: (1,*,*,1) , 3D: (*,*,*,1)
latent_dim = 256
train_ds_path = "/sise/home/lionb/cell_generator/not_masked/image_list_train.csv"
test_ds_path = "/sise/home/lionb/cell_generator/not_masked/image_list_test.csv"