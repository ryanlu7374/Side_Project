import cv2
import numpy as np

color = ((16, 59, 0),(47, 255, 255))
lower = np.array(color[0], dtype="uint8")
upper = np.array(color[1], dtype="uint8")

cap = cv2.VideoCapture(0)
ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

WIDTH = 400
HEIGHT = int(WIDTH / ratio)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (11, 11), 0)
    #使用高斯濾鏡模糊圖像,去除細小顏色差別。
    #原型 dst = cv.GaussianBlur（  src，ksize，sigmaX [，dst [，sigmaY [，borderType]]] ）
    #src 輸入圖像；圖像可以具有任意數量的通道，這些通道可以獨立處理，但深度應為CV_8U，CV_16U，CV_16S，CV_32F或CV_64F。
    #ksize 高斯核大小。ksize.width和ksize.height可以不同，但​​它們都必須為正數和奇數。或者，它們可以為零，然後根據sigma計算得出。
    #sigmaX X方向上的高斯核標準偏差。
    #dst 輸出與src具有相同大小和類型的圖像。
    
    mask = cv2.inRange(hsv, lower, upper)
    #檢查數組元素是否位於其他兩個數組的元素之間，如果src在指定的框內，則dst設置為255（全1位），否則設置為0。。
    #原型 dst =   cv.inRange( src, lowerb, upperb[, dst]  )
    #src 第一個輸入數組。
    #lowerb 包含下邊界數組或標量。
    #upperb 包含上邊界數組或標量。
    #dst 輸出數組，其大小與src和CV_8U類型相同。
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        #根據輪廓區域(key)，獲得最大輪廓。        
        #print("cv2.contourArea",cv2.contourArea)
        #print("contours",contours)
        #print("cnt",cnt)
        if cv2.contourArea(cnt) > 100:
        #計算輪廓區域。
        #原型  retval  =  cv.contourArea( contour[, oriented] )
        #contour 2D點（輪廓頂點）的輸入向量。
                        
            x, y, w, h = cv2.boundingRect(cnt)
            #計算點集或灰度圖像的非零像素的右上點邊界矩形。
            #原型 retval  =  cv.boundingRect(array)
            #array 灰度圖像或2D點集。
            
            x1, y1, x2, y2 = x-2, y-2, x+w+4, y+h+4
            
            out = cv2.bitwise_and(hsv, hsv, mask=mask)
            #計算兩個數組的按位求和（dst = src1＆src2）。
            #原型 dst =   cv.bitwise_and( src1, src2[, dst[, mask]]   )
            #src1 第一個輸入數組或標量。
            #src2 第二個輸入數組或標量。
            #mask 可選操作掩碼，8位單通道數組，用於指定要更改的輸出數組的元素。
            #dst 具有與輸入數組相同的大小和類型的輸出數組。
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.rectangle(hsv, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
            
        frame = cv2.hconcat([frame, hsv, out])
        cv2.imshow("frame", frame)
    if cv2.waitKey(1) == 27:
        cv2.destoryAllWindows()
        break 