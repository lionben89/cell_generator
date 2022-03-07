import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def tf_pearson_corr(y_true, y_pred):
    y_i = tf.where(y_true>0.0)
    x = y_true[y_i]
    y = y_pred[y_i]
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def pearson_corr(a,b):
    a_i = np.argwhere(a > 0.0)
    a_new = a[a_i[:,0],a_i[:,1],a_i[:,2],a_i[:,3]]
    b_new = b[a_i[:,0],a_i[:,1],a_i[:,2],a_i[:,3]]
    
    mean_a = np.mean(a_new)
    mean_b = np.mean(b_new)
    std_a = np.std(a_new - mean_a)
    std_b = np.std(b_new - mean_b)
    cc = np.mean((a_new - mean_a) * (b_new - mean_b)) / (std_a * std_b)
    return cc