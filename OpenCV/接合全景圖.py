import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('img', nargs='+', help='input images')
args = ap.parse_args()

img_arr = []
for filename in args.img:
    image = cv2.imread(filename)
    img_arr.append(image)
    
stitcher = cv2.Stitcher_create()
#創建一個縫合(stitch)配置。
#原型 retval	=cv.Stitcher_create(	[, mode]	)
status, pano = stitcher.stitch(img_arr)
#嘗試拼接給定的圖像。。
#原型 retval, pano	=	cv.Stitcher.stitch(	images[, pano]	)
#images 輸入圖像。
#pano 最終全景圖。

if status == cv2.Stitcher_OK:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', pano)
    cv2.imwrite('final.jpg', pano)
    cv2.waitKey(0)
    cv2.destoryAllWindows()
    print('done')
else:
    print('error: {}'.format(status))