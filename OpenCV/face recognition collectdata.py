import cv2
ESC = 27
# 畫面數量計數
n = 1
# 存檔檔名用
index = 0
# 人臉取樣總數
total = 100

def saveImage(face_image, index):
    filename = 'Face_Recognition/Face_Training_Set/{:03d}.pgm'.format(index)
    cv2.imwrite(filename, face_image)
    print(filename)
    
#載入聯級分類器與開啟攝影機。    
face_cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cv2.namedWindow('video', cv2.WINDOW_NORMAL)

#讀取影像並轉成灰階。
while n > 0:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (600, 400))
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #偵測人臉，並且每5張人臉存檔一次。5張才存一次的目的是讓使用者可以轉一下頭，變化一下表情，讓訓練用的圖片多樣化一點。
    #訓練用圖片的解析度這裡使用 400x400，可自行調整這個數字，但最小為50x50。
    #取樣的時候不要讓畫面中有其他人的人臉被偵測到(因為一次只能標注一人)，
    #這樣會讓訓練用的圖片有雜質干擾，導致之後辨識時準確度降低。
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if n % 5 == 0:
            face_image = cv2.resize(gray[y: y + h, x: x + w], (400, 400))
            saveImage(face_image, index)
            index += 1
            if index >= total:
                print('get training data done')
                n = -1
                cap.release()
                break
        n += 1
        
    #將攝影機拍到的影像顯示出來。
    cv2.imshow('video', frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
