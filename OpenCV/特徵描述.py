import cv2
image = cv2.imread('blox.jpg')

#建立演算法物件
#sift_feature = cv2.xfeatures2d.SIFT_create()
#2020年3月7日後，SIFT的專利將到期，因此應免費使用。
#opencv將其移出非自由文件夾將使所有人都能使用默認標誌編譯opencv並仍然獲得SIFT。
#所以我們再使用SIFT算法，直接這樣寫即可：descriptor = cv2.SIFT_create()
sift_feature = cv2.SIFT_create()
#SIFT算法提取關鍵點
#原型 retval	=cv.SIFT_create(	[, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]]]	)
#使用預設參數值,詳細參考https://docs.opencv.org/master/d7/d60/classcv_1_1SIFT.html
surf_feature = cv2.xfeatures2d.SURF_create()
#SURF算法提取關鍵點https://docs.opencv.org/master/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html
#原型 retval	=cv.xfeatures2d.SURF_create(	[, hessianThreshold[, nOctaves[, nOctaveLayers[, extended[, upright]]]]]	)
#使用預設參數值,詳細參考
orb_feature = cv2.ORB_create()
#ORB關鍵點檢測器和描述符提取器
#原型 retval = cv.ORB_create(  [, nfeatures[, scaleFactor[, nlevels[, edgeThreshold[, firstLevel[, WTA_K[, scoreType[, patchSize[, fastThreshold]]]]]]]]]  )
#使用預設參數值,詳細參考https://docs.opencv.org/master/db/d95/classcv_1_1ORB.html

#找特徵關鍵點
sift_kp = sift_feature.detect(image)
#檢測圖像（第一變體）中的關鍵點。以下兩個算法通用
#原型 keypoints	=	cv.Feature2D.detect(	image[, mask]	)
#image 8位元輸入圖像,灰階。
#mask 指定在哪裡尋找關鍵點的掩碼（可選）。它必須是一個8位整數矩陣，在目標區域中具有非零值。
#keypoints 檢測到的關鍵點。
surf_kp = surf_feature.detect(image)
orb_kp = orb_feature.detect(image)

#畫關鍵點
sift_out = cv2.drawKeypoints(image, sift_kp, None)
#檢測圖像（第一變體）中的關鍵點。以下兩個算法通用
#原型 outImage	=	cv.drawKeypoints(	image, keypoints, outImage[, color[, flags]])
#image 源圖像。
#keypoints 源圖像中的關鍵點。
#outImage 輸出圖像。
surf_out = cv2.drawKeypoints(image, surf_kp, None)
orb_out = cv2.drawKeypoints(image, orb_kp, None)

#圖合併後畫出
#image = cv2.vconcat(cv2.hconcat([image, sift_out]), cv2.hconcat([surf_out, orb_out]))
#image = cv2.vconcat([image, sift_out, surf_out, orb_out])
image = cv2.hconcat([image, sift_out, surf_out, orb_out])
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destoryAllWindows()