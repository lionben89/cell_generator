import global_vars as gv
from gui_logic import *


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

gv.patch_size = (32,128,128,1)
gv.interpert_model_path = "../unet_model_22_05_22_ngc_128"
gv.model_path = "../mg_model_ngc_13_05_24_1.5"
gv.organelle = "Nucleolus-(Granular-Component)" 
#"Plasma-membrane" #"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)"
base_dir = "/sise/assafzar-group/assafzar/full_cells_fovs"
dataset_path = "{}/train_test_list/{}/image_list_test.csv".format(base_dir,gv.organelle)
X_gradcam = False
layer_name = "unet_convt_bottleneck2"

unet_model = load_model(gv.interpert_model_path)
mg_model = load_model(gv.model_path)
dataset = get_dataset(dataset_path)

selected_layer = unet_model.get_layer(layer_name)
evaluate_interperters(gv.model_path,dataset,unet_model,mg_model,selected_layer,X_gradcam)    
plot_evaluation_graph_std("{}/comparison.svg".format(gv.model_path),["saliency","gradcam","mask_interperter"])