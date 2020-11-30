import cv2

image = cv2.imread('50coin2.jpeg', -1)
print('imageshape', image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 0)

edged = cv2.Canny(gray, 20, 40)
#使用Canny算法在輸入圖像中找到邊緣，並在輸出地圖邊緣中對其進行標記。
#原型 edges = cv.Canny( image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]    )
#image 8位元輸入圖像,灰階。
#threshold1 磁滯過程的第一閾值。
#threshold2 磁滯過程的第二個閾值。
#edges 輸出邊緣圖；單通道8位圖像，其大小與image相同。

print('edgedshape', edged.shape)
print('edged', edged)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('contoursshape', len(contours))
#contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print('contoursshape', len(contours))

out = image.copy()
out.fill(0)
cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
image = cv2.hconcat([image, out])
cv2.imshow('frame', image)

cv2.waitKey(0)
cv2.destoryAllWindows()