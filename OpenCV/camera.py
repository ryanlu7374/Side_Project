import cv2
import time

ESC = 27
n, begin_time = 0, time.time()

#####cv2.namedWindow('frame_win', cv2.WINDOW_NORMAL)
#cv2.imshow 所開啟的視窗會依據圖片來自動調整大小，
#希望可以自由縮放視窗的大小，這時候就使用 cv2.namedWindow
#原型cv2.namedWindow(winname[, flags])
#flags:1.WINDOW_NORMAL 用戶可以調整窗口的大小 2.WINDOW_AUTOSIZE 自動調整窗口大小以適合顯示的圖像

cap = cv2.VideoCapture(0)
#原型cv2.VideoCapture(filename|device)
#filename:打開的視頻文件（例如video.avi）或圖像序列的名稱（例如img_％02d.jpg，將讀取img_00.jpg，img_01.jpg等示例）
#device:打開的視頻捕獲設備的ID（即攝像機索引）。如果連接了單個攝像機，則只需傳遞0。

ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#返回指定的VideoCapture屬性
#原型cv2.VideoCapture.get(propId)
#CV_CAP_PROP_FRAME_WIDTH視頻流中幀的寬度。
#CV_CAP_PROP_FRAME_HEIGHT視頻流中幀的高度。

#WD = cv2.CAP_PROP_FRAME_WIDTH
#HG = cv2.CAP_PROP_FRAME_HEIGHT

WIDTH = 600
HEIGHT = int(WIDTH / ratio)

while True:
    ret, frame = cap.read()
    #抓取，解碼並返回下一個視頻幀。
    #原型Python: cv2.VideoCapture.read([image]) → retval, image
    
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    #調整圖像大小。
    #原型Python: cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) → dst
    #src 輸入圖像。dsize 輸出圖像尺寸。dst 輸出圖像。
    
    frame = cv2.flip(frame, 1)
    #圍繞垂直軸，水平軸或兩個軸翻轉2D數組。
    #原型cv2.flip(src, flipCode[, dst]) → dst
    #src 輸入數組。dst 與相同大小和類型的輸出數組src。
    #flipCode 一個指定如何翻轉數組的標誌；0表示繞x軸翻轉，正值（例如1）表示繞y軸翻轉。負值（例如-1）表示繞兩個軸翻轉。
    
    cv2.imshow('frame', frame)
    n += 1
    print('FPS: {:5.2f}'.format(n / (time.time() - begin_time)))
    
    if cv2.waitKey(27) == ESC:
    #原型cv.WaitKey(delay=0) → int
    #delay 延遲（以毫秒為單位）。0是表示“永遠”的特殊值。
    #它返回所按下鍵的代碼；如果在經過指定時間之前未按下任何鍵，則返回-1。
        cap.release()
        #先放掉相機,才能關視窗
        cv2.destroyAllWindows()
        #銷毀所有HighGUI窗口
        break
