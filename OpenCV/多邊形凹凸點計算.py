import cv2

frame = cv2.imread('star.png')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 50, 150)
edged = cv2.dilate(edged, None, iterations=1)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
hull = cv2.convexHull(cnt, returnPoints=False)
#查找點集的凸包。就是找凸
#原型 hull	= cv.convexHull(	points[, hull[, clockwise[, returnPoints]]]	)
#points 輸入二維點集。
#returnPoints 操作標誌。在矩陣的情況下，當標誌為true時，函數將返回凸包點。否則，它返回凸包點的索引。
#approxCurve 輸出凸包。它可以是索引的整數向量，也可以是點的向量。在第一種情況下，外殼元素是原始數組中凸包點的基於0的索引（因為凸包點集是原始點集的子集）。在第二種情況下，外殼元素本身就是凸形外殼點。
defects = cv2.convexityDefects(cnt, hull)
#查找輪廓的凸度缺陷。就是找凹
#原型 convexityDefects	=	cv.convexityDefects(	contour, convexhull[, convexityDefects]	)
#contour 輸入二維點集。
#convexhull 	使用convexHull獲得的凸包，其中應包含構成該包的輪廓點的索引。
#convexityDefects 凸度缺陷的輸出向量。在C ++和新的Python / Java接口中，每個凸度缺陷都表示為4元素整數向量（aka Vec4i）：（start_index，end_index，farthest_pt_index，fixpt_depth），
#                 其中索引是凸度缺陷原始輪廓中基於0的索引起點，終點和最遠點，以及fixpt_depth是最遠輪廓點和船體之間距離的定點近似值（具有8個小數位）。也就是說，要獲取深度的浮點值將為fixpt_depth / 256.0。

print('凸點數量：{}'.format(len(hull)))
print('凹點數量：{}'.format(len(defects)))

cnt = contours[0]
cnt = cv2.approxPolyDP(cnt, 30, True)
hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)
print('凸點數量2：{}'.format(len(hull)))
print('凹點數量2：{}'.format(len(defects)))

for i in range(defects.shape[0]):
    s, e, f, d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(frame, start, end, (0,255,0), 2)
    cv2.circle(frame, far, 5, (0,0,255), -1)
    
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destoryAllWindows()