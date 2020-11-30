import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i1', '--image1', required=True, help='first image')
ap.add_argument('-i2', '--image2', required=True, help='second image')
args = vars(ap.parse_args())

img1 = cv2.imread(args['image1'])
img2 = cv2.imread(args['image2'])

feature = cv2.ORB_create()
kp1, des1 = feature.detectAndCompute(img1, None)
#檢測關鍵點併計算描述符
#原型 keypoints, descriptors	=	cv.Feature2D.detectAndCompute(	image, mask[, descriptors[, useProvidedKeypoints]]	)
#image 8位元輸入圖像,灰階。
#mask 指定在哪裡尋找關鍵點的掩碼（可選）。它必須是一個8位整數矩陣，在目標區域中具有非零值。
#keypoints 檢測到的關鍵點。
kp2, des2 = feature.detectAndCompute(img2, None)

#ORB的距離要使用Hamming distance,SIFT或SURF使用預設NORM_L2
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#蠻力描述符匹配器。
#原型 <BFMatcher object>	=	cv.BFMatcher(	[, normType[, crossCheck]]	)
#使用預設參數值,詳細參考https://docs.opencv.org/master/d3/da1/classcv_1_1BFMatcher.html
matches = bf.match(des1, des2)
#從查詢集中找到每個描述符的k個最佳匹配。
#原型matches	=	cv.DescriptorMatcher.match(	queryDescriptors, trainDescriptors[, mask]	)
#queryDescriptors 查詢描述集。
#trainDescriptors 訓練描述集。該集合不會添加到存儲在類對像中的火車描述符集合中。
#matches matches。如果查詢描述符在mask中被屏蔽，則不會為此描述符添加匹配項。因此，匹配大小可能小於查詢描述符的數量。
print('matches', matches)

matches = sorted(matches, key=lambda x:x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], outImg=None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
#從兩個圖像繪製找到的關鍵點匹配項。
#原型 outImg	=	cv.drawMatches(	img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]	)
#變化型 outImg	=	cv.drawMatchesKnn(	img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]	)
#img1 第一個原始圖片。
#keypoints1 來自第一個源圖像的關鍵點。
#img2 第二個源圖像。
#keypoints2 來自第二個源圖像的關鍵點。
#matches1to2 從第一張圖片匹配到第二張圖片，這意味著keypoints1 [i]在keypoints2 [matches [i]]中具有一個對應點。
#outImg 輸出圖像。
#flags 標誌設置圖形功能。可能的標誌位值由DrawMatchesFlags定義。

width, height, channel = img3.shape
ratio = float(width) / float(height)
img3 = cv2.resize(img3, (1024, int(1024*ratio)))
cv2.imshow('image', img3)
cv2.waitKey(0)
cv2.destoryAllWindows()