import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import np_utils          #for one_hot encoding

import img_list
import img_preprocess
import sys
import gc

samplepath = "D:\\trainpath"

def load_data(GrayorRGB=True, valid_pc=0.18, img_size=224, samplepath=samplepath, label=None):
    
    img_paths, y_train, y_trainOnehot = img_list.imglist(samplepath, label=True)

    """ Train class_weight """
    class_weight = 'balanced'
    classes= np.unique(y_train)
    weight = compute_class_weight(class_weight, classes=classes, y=y_train)
    
    """ Split the training set and validation set """
    X_train_paths, X_valid_paths, y_trainOnehot, y_validOnehot = train_test_split(img_paths, y_trainOnehot, test_size=valid_pc, stratify=y_trainOnehot)
    
    X_valid = img_preprocess.imgpreprocess(GrayorRGB=GrayorRGB, img_size=img_size, img_paths = X_valid_paths)
    X_valid = np.array(X_valid, dtype=np.float32)
    
    return X_train_paths, X_valid, y_trainOnehot, y_validOnehot, weight

def yield_generater(X_train_paths, y_trainOnehot, batch_size, GrayorRGB, img_size):
    count=0
    start=0
    batch_size=batch_size
    while 1:
        if start+batch_size>len(X_train_paths) :
            start=0
        
        X_train = img_preprocess.imgpreprocess(GrayorRGB=GrayorRGB, img_size=img_size, img_paths = X_train_paths[start:start+batch_size], train_enable=True)  
        X_train = np.array(X_train, dtype=np.float32)
        y_trainOnehot_ = y_trainOnehot[start:start+batch_size]
        
        count=count+1
        start=start+batch_size

        yield(X_train, y_trainOnehot_)

        """ Delete memory and recycle """
        del X_train
        del y_trainOnehot_
        gc.collect()

""" Test datas & labels"""
def load_test_data(GrayorRGB=True, img_size=224, samplepath=None, label=None):
    
    img_paths, y_test, y_testOnehot = img_list.imglist(samplepath, label)
    
    X_test = img_preprocess.imgpreprocess(GrayorRGB=True, img_size=img_size, img_paths = img_paths)

    X_test = np.array(X_test, dtype=np.float32)
    
    X_id = []
    for path in img_paths:
        img_id = path.split('/')[-1]
        X_id.append(img_id)
        
    return X_test, X_id, y_test, y_testOnehot