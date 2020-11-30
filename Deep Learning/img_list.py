import os
import glob
import numpy as np
from keras.utils import np_utils

# Build img & label array
def imglist (path_,label=None):
    img_paths = []
    img_labels = []
    img_labelsOnehot = np.array([])
    label_key = 0
    types = ('*.bmp', '*.jpg', '*.png')
    
    
    if (label == False):
        if (os.path.isdir(path)):
            for type_ in types:
                file_img = os.path.join(path, type_)
                for img_path in glob.glob(file_img):
                    img_paths.append(img_path)
            
        else:
            img_paths.append(path)
            
        img_labels = np.array(img_labels, dtype=np.float32)
            
    elif (label == True):
        for folder in os.listdir(path):
            for type_ in types:
                file_img = os.path.join(path, folder, type_)
                for img_path in glob.glob(file_img):
                    img_paths.append(img_path)
                    img_labels.append(label_key)
            label_key += 1
            
        img_labels = np.array(img_labels, dtype=np.float32)
        img_labelsOnehot = np_utils.to_categorical(img_labels)
        
 
    return img_paths, img_labels, img_labelsOnehot