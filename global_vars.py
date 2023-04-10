model_type = "UNETP"
model_path = "./aae_model_ne_27_03_22_128_fl_next" #"./aae_model_ne_27_03_22_128_fl" #"./aae_model_ne_27_03_22_128_fl_next" #"./aae_model_ne_26_03_22_128" #"./aae_model_ne_27_03_22_128_next" #"./aae_model_ne_26_03_22_bf_1024"        #"./aae_model_ne_22_03_22_seg" #"./aae_model_ne_22_03_22_bf"
latent_to_latent_model_path = "./model_ltol_dim_16_256_256_ld_64_config_bf2ne"
unet_model_path = "./unet_model_22_05_22_mito_128"#"./unet_model_x_mse_2_3_nobn_mito_seg_16_04"#unet_seg_model_x_mse_2_3_nobn_ne_seg_13_04 unet_model_x_mse_2_3_nobn_ne_seg_13_04 _mse_2_3_nobn_ne_seg (loss=0.001 lr=0.00001) _mse_2_3_nobn(loss=0.01 lr=0.0001) _mse_2_3_nobn_ne(loss=0.01 lr=0.0001) _mae"./unet_model_06_04_22" #"./unet_model_ne_05_04_22_seg_bce"#"./unet_model_ne_05_04_22_seg_bce" #"./unet_model_ne_04_04_22_seg"
pre_unet_model_path = "./unet_model_x_mse_2_3_nobn_ne"
sg_model_path = "./sg_model_ne_02_04_22"
stylegan_path = "./stylegan"
eam_model_path = "./eam_model_ne_16_04_22_fl_zero" #_baseline_adaptor
shg_model_path = "./ig_model_24_05_22_golgi_128"
zg_model_path = "zg_model_ne_18_04_22_ne_adaptor" #"./zg_model_18_04_22_ngc" #"./zg_model_ne_18_04_22_ne_adaptor"
mg_model_path = "mg_model_mito_10_06_22_5_0_new2"#"mg_model_mito_11_07_22_normal_0.95_b"
rc_model_path = "./rc_model_21_04_22_a"
pm_model_path = "./pm_model_27_04_22_ne_patchs_co"
sn_model_path = "./sn_model_24_04_22_ne_d"
clf_model_path = "./clf_model_14_12_22"
patch_size = (32,128,128,1) ## 2D: (1,*,*,1) , 3D: (*,*,*,1)
latent_dim = 128 #128 #64 #1024
number_epochs = 100 #10000
batch_size = 4
batch_norm = True

# input = "channel_target"
input = "channel_signal"
# input = "structure_seg"
target = "channel_target"
# target = "channel_signal"
# target = "structure_seg"

organelle = "Mitochondria"

train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(organelle)
# train_ds_path = "/sise/home/lionb/cell_generator/image_list_train.csv"
# train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nucleolus-(Granular-Component)/image_list_train.csv" #norm input
# train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nuclear-envelope/image_list_train.csv" #no norm input
# train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Mitochondria/image_list_train.csv" #no norm input
# train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Endoplasmic-reticulum/image_list_train.csv" #no norm input
# train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Microtubules/image_list_train.csv" #no norm input
# train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Plasma-membrane/image_list_train.csv" #no norm input
# train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Golgi/image_list_train.csv"
# train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Desmosomes/image_list_train.csv"
# train_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Actin-filaments/image_list_train.csv"

test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_test.csv".format(organelle)
# test_ds_path = "/sise/home/lionb/cell_generator/image_list_test.csv"
# test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nucleolus-(Granular-Component)/image_list_test.csv"
# test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Nuclear-envelope/image_list_test.csv"
# test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Mitochondria/image_list_test.csv"
# test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Endoplasmic-reticulum/image_list_test.csv"
# test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Microtubules/image_list_test.csv"
# test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Plasma-membrane/image_list_test.csv"
# test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Golgi/image_list_test.csv"
# test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Desmosomes/image_list_test.csv"
# test_ds_path = "/sise/home/lionb/single_cell_training_from_segmentation/Actin-filaments/image_list_test.csv"

