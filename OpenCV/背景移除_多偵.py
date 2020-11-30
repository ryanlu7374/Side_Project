import cv2

bs = cv2.createBackgroundSubtractorMOG2()
#基於高斯混合的背景/前景分割算法。
#先使用cv2.createBackgroundSubtractor建立物件（可在此階段輸入參數），接著便可傳入影像使用apply命令去學習並取得去除背景後的結果。
#四種前/背景分離指令,1.BackgroundSubtractorMOG 2.BackgroundSubtractorMOG2 3.BackgroundSubtractorGMG 4.BackgroundSubtractorKNN
#參考 https://docs.opencv.org/master/d7/df6/classcv_1_1BackgroundSubtractor.html

cap = cv2.VideoCapture(0)
ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

WIDTH = 400
HEIGHT = int(WIDTH / ratio)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.flip(frame, 1)
    
    gray = bs.apply(frame)
    #基於高斯混合的背景/前景分割算法。計算前景mask。
    #原型  fgmask = cv.BackgroundSubtractorMOG2.apply(  image[, fgmask[, learningRate]] )
    #image 下一個frame。浮點frame資料,將在不縮放的情況下使用，並且在範圍內[ 0 ，255 ]
    #learningRate 0到1之間的值指示學習背景模型的速度。負參數值使算法使用一些自動選擇的學習速率。0表示完全不更新背景模型，1表示從最後一幀完全重新初始化背景模型。
    #fgmask 輸出前景mask,為8位二進製圖像。

    mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=10)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #查找二進製圖像中的輪廓。
    #原型 contours, hierarchy =   cv.findContours(    image, mode, method[, contours[, hierarchy[, offset]]]  )
    #image 源，一個8位單通道圖像。非零像素被視為1。零像素保持為0，因此圖像被視為binary。您可以使用compare，inRange，threshold，adaptiveThreshold，Canny和其他圖像來創建灰度或彩色二進製圖像。如果mode等於RETR_CCOMP或RETR_FLOODFILL，則輸入也可以是標籤的32位整數圖像（CV_32SC1）。
    #mode 輪廓檢索模式，請參閱RetrievalModes。
        #cv.RETR_EXTERNAL 僅檢索極端的外部輪廓。
        #cv.RETR_LIST 在不建立任何層次關係的情況下檢索所有輪廓。
        #cv.RETR_CCOMP 檢索所有輪廓並將其組織為兩級層次結構。在頂層，組件具有外部邊界。在第二層，有孔的邊界。如果所連接組件的孔內還有其他輪廓，則該輪廓仍將放置在頂層。
        #cv.RETR_TREE 檢索所有輪廓，並重建嵌套輪廓的完整層次。
    #method 輪廓近似方法，請參見ContourApproximationModes。
        #cv.CHAIN_APPROX_NONE 絕對存儲所有輪廓點。
        #cv.CHAIN_APPROX_SIMPLE 壓縮水平，垂直和對角線段，僅保留其端點。例如，一個直立的矩形輪廓編碼有4個點。
    #contours 檢測到的輪廓。每個輪廓都存儲為點的向量（例如std :: vector <std :: vector <cv :: Point>>）。
    #hierarchy 可選的輸出向量（例如std :: vector <cv :: Vec4i>），包含有關圖像拓撲的信息。它具有與輪廓數量一樣多的元素。
        #對於每個第i個輪廓輪廓[i]，將元素等級[i] [0]，等級[i] [1]，等級[i] [2]和等級[i] [3]設置為0-在相同的層次級別上，基於下一個和上一個輪廓的輪廓的索引，分別是第一個子輪廓和父輪廓。如果對於輪廓i，沒有下一個，上一個，父級或嵌套的輪廓，則hierarchy [i]的相應元素將為負。
    
    for c in contours:
        if cv2.contourArea(c) < 200:
        #計算輪廓區域。
        #原型 retval = cv.contourArea( contour[, oriented] )
        #contour Input vector of 2D points (contour vertices), stored in std::vector or Mat.
        #oriented 定向區域標誌。如果為true，則函數將根據輪廓方向（順時針或逆時針）返回帶符號的區域值
            continue
        cv2.drawContours(frame, contours, -1,(0, 255, 255), 2)
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.hconcat([frame, mask])
    #將水平串聯應用於給定矩陣。。
    #原型 dst =  cv.hconcat（ src [，dst]  ）
    #src 輸入矩陣的矩陣或向量。所有矩陣必須具有相同的行數和相同的深度。
    #dst 輸出數組。它具有與src相同的行數和深度，以及src的cols之和。
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == 27:
        cv2.destoryAllWindows()
        break    