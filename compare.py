import global_vars as gv
from gui_logic import *


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

gv.patch_size = (32,128,128,1)
gv.unet_model_path = "/sise/home/lionb/unet_model_22_05_22_ngc_128"
gv.mg_model_path = "/sise/home/lionb/mg_model_ngc_10_06_22_5_0_new"
gv.organelle = "Nucleolus-(Granular-Component)" #"Tight-junctions" #Actin-filaments" #"Golgi" #"Microtubules" #"Endoplasmic-reticulum" 
#"Plasma-membrane" #"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)"
dataset_path = "/sise/home/lionb/single_cell_training_from_segmentation/{}/image_list_train.csv".format(gv.organelle)
X_gradcam = False
layer_name = "unet_convt_bottleneck2"

unet_model = load_model(gv.unet_model_path)
mg_model = load_model(gv.mg_model_path)
dataset = get_dataset(dataset_path)

selected_layer = unet_model.get_layer(layer_name)
evaluate_interperters(gv.mg_model_path,dataset,unet_model,mg_model,selected_layer,X_gradcam)    
plot_evaluation_graph_std("{}/comparison.svg".format(gv.mg_model_path),["saliency","gradcam","mask_interperter"])