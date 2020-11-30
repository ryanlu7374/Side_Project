import cv2
import numpy as np
import random

def imgpreprocess(GrayorRGB, img_size, img_paths, train_enable=False):
    img_preprocess = []
    random_brightness=random.randint(0,128)
    
    for path in img_paths:
        if GrayorRGB == False:
            img = cv2.imread(path, 0) #Gray，2D
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) #2D change into 3D
            
        elif GrayorRGB == True:
            img = cv2.imread(path)    #Color，3D            

        img = cv2.resize(img, (img_size, img_size))

        if train_enable == True:
        #brightness
        #rotate 
            random_rotation=random.randint(0,100)
            if random_rotation <= 30:            
                rows,cols,channels = img.shape
                M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
                img = cv2.warpAffine(img,M,(cols,rows))
        
        """ Feature scaling """
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        img_preprocess.append(img)
    
    return img_preprocess