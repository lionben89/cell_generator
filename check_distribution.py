from turtle import color
import tensorflow as tf
import tensorflow.keras as keras
from dataset import DataGen
from cell_imaging_utils.image.image_utils import ImageUtils
import global_vars as gv
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from scipy import ndimage
from models.AAE import *


tf.compat.v1.enable_eager_execution()

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
num_samples=250


def plot_distribution_samples(decoder):
    slice_i = 8
    num_samples_per_feature = 3
    plt.rcParams["figure.figsize"] = (num_samples_per_feature*2,gv.latent_dim*2)
    generated_samples = []
    latent_samples = np.zeros(shape=(num_samples_per_feature,gv.latent_dim,gv.latent_dim))
    sigmas = np.linspace(-2,2,num_samples_per_feature)
    for i in range(len(sigmas)): 
        latent_samples[i,:] = np.diag(np.ones(shape=(gv.latent_dim)))*sigmas[i]
        temp_generated = np.zeros(shape=(gv.latent_dim,*gv.patch_size[1:-1]))
        for j in range(gv.latent_dim):
            temp_generated[j] = decoder(latent_samples[i,j:j+1])[:,slice_i,:,:,0].numpy()
        generated_samples.append(temp_generated)
        
    fig, axs = plt.subplots(gv.latent_dim,num_samples_per_feature,sharex=True,sharey=True)
    for i in range(axs.shape[1]):
        column_images = generated_samples[i]
        for j in range(axs.shape[0]):
            image = column_images[j]#ndimage.zoom(column_images[j],[0.25,0.25],mode="constant",cval=0,order=1,prefilter=False)
            axs[j,i].imshow(image, cmap=plt.gray())
            # axs[j,i].set_axis_off()
            axs[j,i].set_xlabel("mu={}".format(sigmas[i]))
            axs[j,i].set_ylabel("feature={}".format(j))
            
    
    plt.tight_layout()
    plt.savefig("{} images distribuation.png".format(gv.model_path))
    plt.clf()
    

def plot_distribuation_graph(samples):
    plt.rcParams["figure.figsize"] = (64,64)
    y_normal = sample_distribution(num_samples,gv.latent_dim).numpy()
    fig, axs = plt.subplots(int(np.sqrt(gv.latent_dim)),int(np.sqrt(gv.latent_dim)))
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            l = int(i*np.sqrt(gv.latent_dim)+j)
            x_samples = np.linspace(min(samples[:,l]),max(samples[:,l]),num_samples)
            x_normal = np.linspace(min(y_normal[:,l]),max(samples[:,l]),num_samples)
            kde_samples = stats.gaussian_kde(samples[:,l])
            kde_normal = stats.gaussian_kde(y_normal[:,l])
            axs[i,j].hist(samples[:,l],100,density=True,facecolor='r', alpha=0.25)
            axs[i,j].hist(y_normal[:,l],100,density=True,facecolor='g', alpha=0.25)
            axs[i,j].plot(x_samples,kde_samples(x_samples),color='r')
            axs[i,j].plot(x_normal,kde_normal(x_normal),color='g')
            axs[i,j].set_title("{}".format(l))
    plt.tight_layout()
    plt.savefig("{} latent dim distribuation.png".format(gv.model_path))
    plt.clf()
    
def plot_tsne_graph(samples):
    plt.rcParams["figure.figsize"] = (16,16)
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(samples)
    plt.scatter(tsne_results[:,0],tsne_results[:,1])
    plt.xlabel("tsne-1")
    plt.ylabel("tsne-2")
    plt.title("TSNE 2d reduction")
    plt.tight_layout()
    plt.savefig("{} tsne plot.png".format(gv.model_path))
    plt.clf()
    
def plot_correlation_matrix(samples):
    plt.rcParams["figure.figsize"] = (32,32)
    df = pd.DataFrame(samples)
    correlation_mat = df.corr()
    sns.heatmap(correlation_mat, annot = True)
    plt.tight_layout()
    plt.savefig("{} correlation matrix.png".format(gv.model_path))
    plt.clf()
    
def plot_correlation_one_to_rest(samples):
    plt.rcParams["figure.figsize"] = (32,32)
    df = pd.DataFrame(samples)
    fig, axs = plt.subplots(int(np.sqrt(gv.latent_dim)),int(np.sqrt(gv.latent_dim)))
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            l = int(i*np.sqrt(gv.latent_dim)+j)
            correlation_mat = df.corrwith(df[l])
            axs[i,j]=sns.heatmap(np.expand_dims(correlation_mat,axis=-1), annot = True)
    plt.tight_layout()
    plt.savefig("{} correlation matrix one to many.png".format(gv.model_path))
    plt.clf()
    

def generate_samples(num_samples):
    dataset = DataGen(gv.train_ds_path,gv.input,gv.target,batch_size=1,num_batches=num_samples,patch_size=gv.patch_size)
    samples = np.zeros(shape=(num_samples,gv.latent_dim))

    for i in range(num_samples):
        item = dataset.__getitem__(i)
        input_image = item[0]
        target_image = item[1]
        
        z = aae.encoder(input_image).numpy()
        samples[i,:]=z
    return samples

aae = keras.models.load_model(gv.model_path)

# plot_correlation_one_to_rest(samples)
plot_distribution_samples(aae.decoder)

