import cv2
cap = cv2.VideoCapture(0)


trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW',
                'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    # https://docs.opencv.org/master/d0/d0a/classcv_1_1Tracker.html
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
        # https://docs.opencv.org/master/d2/da2/classcv_1_1TrackerCSRT.html
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
   
    return tracker

tracker = createTrackerByName('BOOSTING')
roi = None

while True:
    
    ret, frame =cap.read()
    if roi is None:
        roi = cv2.selectROI('frame', frame)
        #在給定圖像上選擇ROI。功能創建一個窗口，並允許用戶使用鼠標選擇ROI。使用space或enter完成選擇，使用鍵c取消選擇。
        #原型	cv2.selectROI(	windowName, img[, showCrosshair[, fromCenter]])
        #windowName 顯示選擇過程的窗口的名稱。。
        #img 以選擇圖像上ROI。

        if roi != (0, 0, 0, 0):
            tracker.init(frame, roi)
        
    success, rect = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(i) for i in rect]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0, 2))
            
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        cv2.destoryAllWindow()
        break