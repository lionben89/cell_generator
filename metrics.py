import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from math import log10, sqrt

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def tf_pearson_corr_aux(x,y):
    mean_x = tf.reduce_mean(x)
    mean_y = tf.reduce_mean(y)
    std_x = tf.math.reduce_std(x-mean_x)
    std_y = tf.math.reduce_std(y-mean_y)
    cc = tf.reduce_mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)
    return cc

def tf_pearson_corr(y_true, y_pred, weights=None):
    # y_i = tf.where(y_true!=0.0)

    
    if weights is not None:
        ind = tf.where(tf.logical_or(tf.reshape(weights,[-1])==1.0,tf.reshape(weights,[-1])==255.0))
        non_ind = tf.where(tf.logical_and(tf.reshape(weights,[-1])!=1.0,tf.reshape(weights,[-1])!=255.0))    
        cc = 0
        t_weights = [1.0,0.0]
        t = [ind,non_ind]
        for i in range(2):
            [x,y] = tf.cond(tf.shape(ind)[0]>0, lambda: [tf.gather(tf.reshape(y_true,[-1]), t[i]),tf.gather(tf.reshape(y_pred,[-1]), t[i])], lambda: [y_true,y_pred])       
            cc = cc + t_weights[i]*tf_pearson_corr_aux(x,y)
    else:
        x = y_true
        y = y_pred
        cc = tf_pearson_corr_aux(x,y)
    return cc

def pearson_corr(a,b):
    a_i = np.argwhere(a != 1e-4)
    a_new = a[a_i[:,0],a_i[:,1],a_i[:,2],a_i[:,3]]
    b_new = b[a_i[:,0],a_i[:,1],a_i[:,2],a_i[:,3]]
    
    mean_a = np.mean(a_new,dtype=np.float64)
    mean_b = np.mean(b_new,dtype=np.float64)
    std_a = np.std(a_new - mean_a,dtype=np.float64)
    std_b = np.std(b_new - mean_b,dtype=np.float64)
    cc = np.mean((a_new - mean_a) * (b_new - mean_b)) / (std_a * std_b)
    return cc

def dice(a,b):
    k=1
    return np.sum(b[a==k])*2.0 / (np.sum(b) + np.sum(a) + 0.0001)

