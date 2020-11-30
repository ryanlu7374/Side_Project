import cv2
model = cv2.face.LBPHFaceRecognizer_create()
model.read('Face_Recognition/Model/faces.data')
#載入人臉辨識模型。
#原型Python: None = cv.face_FaceRecognizer.read(filename)

print('load training data done')

face_cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
# 可識別化名稱
names = ['LYY']

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (600, 400))
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        face_image = cv2.resize(gray[y: y + h, x: x + w], (400, 400))
        try:
            val = model.predict(face_image)
            #載入人臉辨識模型。
            #原型Python: label, confidence = cv.face_FaceRecognizer.predict(src)
            #src 從中獲取預測的樣本圖像。
            #label 圖像的預測標籤。
            #confidence 預測標籤的相關信心度（例如，距離）。

            print('label:{}, confidence:{}'.format(val[0], val[1]))
            if val[1] < 50:
                cv2.putText(frame, names[val[0]], (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0), 3, cv2.LINE_AA)
        except:
            continue
    cv2.imshow('video', frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break