import cv2

RECT, HEXAGON = 2, 3
frame = cv2.imread('poly.png')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 50, 100)
#使用Canny算法在輸入圖像中找到邊緣，並在輸出地圖邊緣中對其進行標記。
#原型 edges = cv.Canny( image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]    )
#image 8位元輸入圖像,灰階。
#threshold1 磁滯過程的第一閾值。
#threshold2 磁滯過程的第二個閾值。
#edges 輸出邊緣圖；單通道8位圖像，其大小與image相同。
edged = cv2.dilate(edged, None, iterations=1)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print('===處理前')
print('矩形點數量：{}'.format(len(contours[RECT])))
print('六邊形點數量：{}'.format(len(contours[HEXAGON])))
print('contours數量：{}'.format(len(contours)))

approx_rect = cv2.approxPolyDP(contours[RECT], 30, True)
#以指定的精度逼近多邊形曲線。
#原型 approxCurve	=	cv.approxPolyDP(	curve, epsilon, closed[, approxCurve]
#curve 2D點的輸入向量。
#epsilon 指定近似精度的參數。這是原始曲線與其近似值之間的最大距離。
#approxCurve 近似結果。類型應與輸入曲線的類型匹配。
#closed 如果為true，則近似曲線是閉合的（其第一個和最後一個頂點已連接）。否則，它不會關閉。
approx_hex = cv2.approxPolyDP(contours[HEXAGON], 30, True)

print('===處理後')
print('矩形點數量：{}'.format(len(approx_rect)))
print('六邊形點數量：{}'.format(len(approx_hex)))

cv2.drawContours(frame, [approx_rect], -1, (0, 0, 255), 5)
cv2.drawContours(frame, [approx_hex], -1, (0, 0, 255), 5)

cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destoryAllWindows()