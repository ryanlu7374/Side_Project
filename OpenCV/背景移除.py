import cv2
cap = cv2.VideoCapture(0)
#儲存背景
bg = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #將圖像從一種顏色空間轉換為另一種顏色空間。
    #在從RGB顏色空間轉換的情況下，應明確指定通道的順序（RGB或BGR）。
    #請注意，OpenCV中的默認顏色格式通常稱為RGB，但實際上是BGR（字節是相反的）。
    #R，G和B通道值的常規範圍是：CV_8U圖像為0至255.CV_16U圖像為0至65535.CV_32F圖像為0到1
    #原型 dst = cv.cvtColor(src, code[, dst[, dstCn]])
    #src 輸入圖像：8位無符號，16位無符號（CV_16UC ...）或單精度浮點。
    #code 顏色空間轉換代碼（請參閱ColorConversionCodes）。
    #dst 輸出與src具有相同大小和深度的圖像。
    gray = cv2.GaussianBlur(gray, (17, 17), 0)
    #使用高斯濾鏡模糊圖像。
    #原型 dst = cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
    #src 輸入圖像；圖像可以具有任意數量的通道，這些通道可以獨立處理，但深度應為CV_8U，CV_16U，CV_16S，CV_32F或CV_64F。
    #ksize 高斯核大小。ksize.width和ksize.height可以不同，但它們都必須為正數和奇數。或者，它們可以為零，然後根據sigma計算得出。
    #sigmaX X方向上的高斯核標準偏差
    #dst 輸出與src具有相同大小和深度的圖像。
    
    if bg is None:
        bg = gray
        continue
    
    diff = cv2.absdiff(gray, bg)
    #計算兩個數組之間或數組與標量之間的每個元素的絕對差。
    #原型 dst = cv.absdiff( src1, src2[, dst])
    #src1 第一個輸入數組或標量。
    #src2 第二個輸入數組或標量。
    #dst 輸出與src具有相同大小和深度的圖像。
    diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    #將固定級別閾值應用於每個數組元素。
    #該功能通常用於從灰度圖像中獲取一個二級（二進制）圖像（比較也可以用於此目的）或用於去除噪聲，即濾除具有太小或太大值的像素。該功能支持幾種類型的閾值處理。
    #原型 retval, dst =   cv.threshold( src, thresh, maxval, type[, dst]
    #src 輸入數組（多通道，8位或32位浮點）。
    #thresh 閾值。。
    #maxval THRESH_BINARY和THRESH_BINARY_INV閾值類型使用的最大值。
    #type 閾值類型（請參閱ThresholdTypes）。
    #dst 與src具有相同大小，類型和相同通道數的輸出數組。
    diff = cv2.erode(diff, None, iterations=2)
    #該函數使用指定的結構化元素腐蝕源圖像，該結構化元素確定在其上採用最小值的像素鄰域的形狀：
    #原型 dst =   cv.erode(   src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]] )
    #src 輸入圖像；通道數可以是任意的，但深度應為CV_8U，CV_16U，CV_16S，CV_32F或CV_64F之一。
    #kernel 用於侵蝕的結構元素；如果為element=Mat()，3 x 3則使用矩形結構元素。可以使用getStructuringElement創建內核。
    #anchor 錨在元素內的位置；默認值（-1，-1）表示錨點位於元素中心。
    #iterations 施加腐蝕的次數。。
    #dst 與src具有相同大小，類型和相同通道數的輸出數組。
    diff = cv2.dilate(diff, None, iterations=2)
    
    cnts, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    
    for c in cnts:
        if cv2.contourArea(c) < 500:
        #計算輪廓區域。
        #原型 retval = cv.contourArea( contour[, oriented] )

        #contour Input vector of 2D points (contour vertices), stored in std::vector or Mat.
        #oriented 定向區域標誌。如果為true，則函數將根據輪廓方向（順時針或逆時針）返回帶符號的區域值
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        
    cv2.imshow("frame", diff)
    if cv2.waitKey(1) == 27:
        cv2.destoryAllWindows()
        break