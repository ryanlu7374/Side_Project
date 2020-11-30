import cv2
#type_tuple
RECT = ((220, 20), (370, 190))
(left, top), (right, bottom) = RECT

def roiarea(frame):
    return frame[top:bottom, left:right]

def replaceroi(frame, roi):
    frame[top:bottom, left:right] = roi
    return frame

cap = cv2.VideoCapture(0)
ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

WIDTH = 400
HEIGHT = int(WIDTH / ratio)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.flip(frame, 1)
    
    #取出子畫面
    roi = roiarea(frame)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #將處理完的子畫面貼回到原本畫面中
    #其實不用replaceroi,因為陣列資料本來就是傳址，所以cvtColor後原本的畫面就已經修改。
    #除非貼到另個畫面才要replaceroi
    frame = replaceroi(frame, roi)
    
    #在ROI範圍處畫個框
    cv2.rectangle(frame, RECT[0], RECT[1], (0,0,128), 2)
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break