import cv2
import numpy as np

images = []
labels = []
for index in range(100):
    filename = 'Face_Recognition/Face_Training_Set/{:03d}.pgm'.format(index)
    print('read ' + filename)
    img = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    images.append(img)
    # 第一張人臉的標籤為 0(因為一次只能標注一人,多人分開標注一起訓練)
    labels.append(0)
    
print('training...')
model = cv2.face.LBPHFaceRecognizer_create()
#循環局部二值化特徵（用於訓練和預測）只能訓練灰度圖。
#原型Python: retval = cv.face.LBPHFaceRecognizer_create([, radius[, neighbors[, grid_x[, grid_y[, threshold]]]]])

model.train(np.asarray(images), np.asarray(labels))
#使用給定的數據和標籤訓練。
#原型Python: None = cv.face_FaceRecognizer.train(src, labels)
#src 訓練圖像，要學習的面孔。數據以vector的形式。
#labels 與圖像相對應的標籤。數據以vector<int>的形式 或CV_32SC1類型的Mat形式。

model.save('Face_Recognition/Model/faces.data')
#儲存訓練後模型。

print('training done')