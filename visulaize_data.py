import os
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from dataset import DataGen
import global_vars as gv

if (not os.path.exists("./visualizations")):
    os.makedirs("./visualizations")
    
#Projection methods
def get_target_patchs_from_pairs(batch):
    return batch[0][1]

#Process methods
def get_average(batch):
    return np.average(batch,axis=(1,2,3,4))

def get_max(batch):
    return np.max(batch,axis=(1,2,3,4))

#Visulaztion methods
def histogram(data,args,title):
    n, bins, patches = plt.hist(data, **args)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title(title)
    plt.grid(True)
    plt.savefig("./visualizations/{}.png".format(title))

num_iterations = 10
pipline = {"name":"Max Nuclear envelope value in patch","batch_projection":get_target_patchs_from_pairs,"batch_process":get_max,"visulaize_method":lambda x:histogram(x,{"bins":20, "density":True, "facecolor":'b', "alpha":0.75, "cumulative":True},"Max Nuclear envelope value in patch")}

aggregated_values = []
print("Running pipline: {}".format(pipline["name"]))
for i in range(num_iterations):
    print ("Iteration {}/{}".format(i+1,num_iterations))
    train_dataset = DataGen(gv.train_ds_path ,gv.input,gv.target,batch_size = gv.batch_size, num_batches = 32, patch_size=gv.patch_size,min_precentage=0,max_precentage=1,augment=False,pairs=True,neg_ratio=0)
    n = train_dataset.__len__()
    for j in range(n):
        batch = train_dataset.__getitem__(j)
        projected_batch = pipline["batch_projection"](batch)
        processed_batch = pipline["batch_process"](projected_batch)
        aggregated_values = np.concatenate([aggregated_values,processed_batch],axis=0)
pipline["visulaize_method"](aggregated_values)
    

