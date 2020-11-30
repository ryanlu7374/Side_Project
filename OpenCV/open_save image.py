import cv2, time
from datetime import datetime

#Open image file##########################
WIDTH = 1200

frame = cv2.imread('01.jpeg', cv2.IMREAD_UNCHANGED)
#讀取圖像。
#原型Python: cv2.imread(filename[, flags]) → retval
#flags 指定讀取圖像的顏色類型: (詳細參考官網)
#IMREAD_UNCHANGED 按原樣讀取圖像（帶有Alpha通道，否則將被裁剪）。
#IMREAD_COLOR 將圖像轉換為3通道BGR彩色圖像。
#IMREAD_GRAYSCALE 將圖像轉換為單通道灰度圖像。
#IMREAD_ANYDEPTH 當輸入具有相應的深度時返回16位/ 32位圖像，否則將其轉換為8位。

size = frame.shape
ratio = frame.shape[0]/frame.shape[1]
HEIGHT = int(WIDTH * ratio)
frame = cv2.resize(frame, (WIDTH, HEIGHT))
cv2.imshow('image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Save image file##########################
ESC = 27
#IMWRITE_PNG_COMPRESSION
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(27) == ESC:
        if ret:
            cv2.imwrite("{}.png".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            #儲存圖像。通常只能保存8位單通道或3通道（具有“ BGR”通道順序）圖像，但以下情況除外：
            #PNG，JPEG 2000和TIFF格式，可以保存16位無符號（CV_16U）圖像。
            #TIFF，OpenEXR和Radiance HDR格式保存32位浮點（CV_32F）圖像
            #可以保存帶有Alpha通道的PNG圖像。為此創建8位（或16位）4通道圖像BGRA，其中alpha通道位於最後。
            #完全透明的像素應將alpha設置為0，完全不透明的像素應將alpha設置為255/65535。
            #對於PNG，第三個參數表示的是壓縮等級。cv2.IMWRITE_PNG_COMPRESSION，從0到9，壓縮等級尺寸，圖像尺寸越小。尺寸等級為3。
            #對於JPEG，其表示的是圖像的質量，用0-100的整數表示，替換為95。注意，cv2.IMWRITE_JPEG_QUALITY類型為Long，必須轉換成int。
            #原型Python: retval = cv2.imwrite( filename, img[, params]
            #filename 文件名。
            #img 一個或多個圖像被保存。        
       
        else:
            print(' 讀取影像失敗 ')
        cap.release()
        cv2.destroyAllwindow()
        break

