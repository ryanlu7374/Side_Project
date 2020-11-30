import cv2
import numpy as np

gc = np.zeros((512, 512, 3), dtype=np.uint8)
gc.fill(255)
size = gc.shape

#畫直線
cv2.line(gc, (10, 50),(400, 300),(255, 0, 0),5)
cv2.line(gc, (10, 50),(400, 300),(255, 0, 0),5,shift=1)
#原型Python: img = cv.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
#img 圖片。
#pt1 線段的第一點。
#pt2 線段的第二點。
#color 線條顏色。
#thickness 線的粗細。
#lineType 線的類型。請參見LineTypes。
#shift 點坐標中的小數位數。點坐標只能用整數,靠shift給小數位數
cv2.imshow('draw', gc)
cv2.waitKey(0)
cv2.destroyAllWindows()

#畫矩型
gc.fill(255)
cv2.rectangle(gc, (30, 50),(200, 280),(0, 0, 255),5)
cv2.rectangle(gc, (100, 200),(196, 276),(234, 151, 102),-1)
#原型Python: img = cv.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
#img 圖片。
#pt1 矩形的頂點。
#pt2 與pt1相反的矩形的頂點。
#color 矩形的顏色或亮度（灰度圖像）。
#thickness 組成矩形的線的粗細。負值表示該繪製填充矩形。
#lineType 線的類型。請參見LineTypes。
#shift 點坐標中的小數位數。
cv2.imshow('draw', gc)
cv2.waitKey(0)
cv2.destroyAllWindows()

#畫圓
gc.fill(255)
cv2.circle(gc, (200, 100),80,(255, 255, 0),-1)
cv2.circle(gc, (280, 180),60,(147, 113, 217),3)
#原型Python: img = cv.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
#img 圖片。
#center 圓心。
#radius 圓的半徑。
#color 圓圈顏色。
#thickness 圓形輪廓的粗細（如果為正）。負值表示要繪製實心圓。
#lineType 圓邊界的類型。請參見LineTypes。
#shift 中心坐標和半徑值中的小數位數。
cv2.imshow('draw', gc)
cv2.waitKey(0)
cv2.destroyAllWindows()

#畫橢圓
gc.fill(255)
cv2.ellipse(gc, (200, 100),(80, 40),45,0,360,(80, 127, 255),5)
cv2.ellipse(gc, (250, 200),(70, 70),0,0,135,(44, 141, 108),-1)
#原型Python: img = cv.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]])
#img 圖片。
#center 橢圓的中心。
#axes 長短軸長度。
#angle 橢圓旋轉角度，以度為單位。
#startAngle 橢圓弧的起始角度，以度為單位。
#endAngle 橢圓弧的終止角度，以度為單位。
#color 橢圓色。
#thickness 橢圓弧輪廓的粗細（如果為正）。否則將繪製填充橢圓扇形。
#lineType 橢圓邊界的類型。請參見LineTypes。
#shift 中心坐標和軸值的小數位數。
cv2.imshow('draw', gc)
cv2.waitKey(0)
cv2.destroyAllWindows()

#畫多邊型
gc.fill(255)
pts = np.array(((10,5), (100,100), (170,120), (200,50)))
# True: 頭尾相連 ; False: 頭尾不相連
cv2.polylines(gc, [pts], False, (105, 105, 105), 2)
#原型Python: img = cv.polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]])
#img 圖片。
#pts 多邊形頂點組。
#isClosed 頭尾點是否相連。
#color 折線顏色。
#thickness 折線邊緣的厚度。
#lineType 線段的類型。請參見LineTypes
#shift  頂點坐標中的小數位數。
cv2.imshow('draw', gc)
cv2.waitKey(0)
cv2.destroyAllWindows()

#顯示文字
gc.fill(255)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(gc, 'OpenCV', (10, 200), font, 3.5, (255,0,0), 4, cv2.LINE_AA)
cv2.putText(gc, 'OpenCV', (10, 300), font, 3.5, (255,0,0), 4, cv2.LINE_AA, bottomLeftOrigin=True)
#原型Python: img = cv.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
#img 圖片。
#text 要繪製的文字。
#org 文本字在圖像中的左下角座標。
#fontFace 字體類型，請參見HersheyFonts。
#fontScale 字體比例因子乘以特定於字體的基本大小。
#color 文字顏色。
#thickness 用於繪製文本的線條的粗細。
#lineType 線型。查看線型
#bottomLeftOrigin 如果為true，則圖像數據原點位於左下角。否則，它位於左上角。
cv2.imshow('draw', gc)
cv2.waitKey(0)
cv2.destroyAllWindows()