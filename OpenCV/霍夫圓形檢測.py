import cv2
import numpy as np

src = cv2.imread('cup2.jpg', -1)
print('srcshape', src.shape)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, None, 10, 100, 100, 200)
#使用霍夫變換在灰度圖像中查找圓。
#原型 circles	= cv.HoughCircles(	image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]	)
#image 8位單通道灰度輸入圖像。
#method 檢測方法，1.cv.HOUGH_GRADIENT 2.cv.HOUGH_GRADIENT_ALT
#dp 累加器分辨率與圖像分辨率的反比。例如，如果dp = 1，則累加器具有與輸入圖像相同的分辨率。
#   如果dp = 2，則累加器的寬度和高度是其一半。對於HOUGH_GRADIENT_ALT，推薦值為dp = 1.5，除非需要檢測到一些很小的圓圈。
#minDist 檢測到的圓心之間的最小距離。如果參數太小，則除了真實的圓圈外，還可能會錯誤地檢測到多個鄰居圓圈。如果太大，可能會錯過一些圓圈。
#circles 
#param1 第一個方法特定的參數。
#       對於HOUGH_GRADIENT和HOUGH_GRADIENT_ALT，這是傳遞給Canny邊緣檢測器的兩個閾值中的較高閾值（較低的閾值是較小的兩倍）。
#       請注意，HOUGH_GRADIENT_ALT使用Scharr算法來計算圖像導數，因此閾值範圍通常較高，例如300或正常曝光和對比圖像。
#param2 第二種方法特定的參數。
#       在HOUGH_GRADIENT的情況下，它是檢測階段圓心的累加器閾值。它越小，可能會檢測到更多的假圓圈。與較大的累加器值相對應的圓將首先返回。
#       對於HOUGH_GRADIENT_ALT算法，這是圓形的“完美”度量。它越接近1，則選擇的形狀更好的圓形算法。
#       在大多數情況下，0.9應該可以。如果要更好地檢測小圓圈，可以將其降低到0.85、0.8甚至更低。
#       但是，然後還要嘗試限制搜索範圍[minRadius，maxRadius]，以避免出現許多錯誤的圓圈。
#minRadius 最小圓半徑。
#maxRadius 最大圓半徑。如果<= 0，則使用最大圖像尺寸。如果<0，則HOUGH_GRADIENT返回中心而未找到半徑。HOUGH_GRADIENT_ALT始終計算圓半徑。

if len(circles) > 0:
    out = src.copy()
    for x, y, r in circles[0]:
        #畫圓
        cv2.circle(out, (x, y), int(r), ( 0, 0, 255), 3, cv2.LINE_AA)
        #畫圓心
        cv2.circle(out, (x, y), 2, ( 0, 255, 0), 3, cv2.LINE_AA)
        
    src = cv2.hconcat([src, out])
    
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', src)
cv2.waitKey(0)
cv2.destoryAllWindows()