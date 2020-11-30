import cv2
face_cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_eye.xml')
#error: (-215:Assertion failed) !empty() in function 'detectMultiScale'
#找不到haarcascade_frontalface_alt2.xml,要指定路徑
image = cv2.imread('face01.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
size = gray.shape
faces = face_cascade.detectMultiScale(gray, 1.07, 4, minSize=(30, 30), maxSize=(80, 80))
#原型Python: objects = cv.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]])
#img 包含要檢測對象的圖像矩陣。
#scaleFactor 每個圖像縮小多少比例。
#(??,書上說尋找過程中每次檢測時檢測窗口的放大級距，最小值必須為 1。
#此數值越大，可能造成圖片中有人臉但無法偵測出來，反之數值越小越容易偵測到人臉，但偵測速度較慢)
#minNeighbors 每個候選矩形必須保留多少個鄰居。
#(??,書上說需通過幾次檢測才算是人臉。)
#minSize 最小可能的對像大小。小於此值的對象將被忽略。
#maxSize 最大可能的對像大小。大於此值的對象將被忽略。如果maxSize == minSize模型以單一比例評估
for (x, y, w, h) in faces:
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    face_rect = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_rect, 1.05, 8)
    for (ex, ey, ew, eh) in eyes:
        center = (x + ey + int(ew / 2.0), y + ey + int(eh / 2.0))
        r = int(min(ew, eh) / 2.0)
        image = cv2.circle(image, center, r, (255, 255, 0), 5)
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.imshow('video', image)
cv2.waitKey(0)
cv2.destroyAllWindows()