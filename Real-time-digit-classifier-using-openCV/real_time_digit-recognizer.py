
import cv2
import numpy as np
from keras.models import load_model
import pandas as pd
width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

model = load_model('trained2.h5')

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (28,28))
    img = preProcessing(img)
    cv2.imshow("duh", img)
    img = img.reshape(1, 28, 28, 1)
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    cv2.putText(imgOriginal, str(np.argmax(predictions,axis=1)) + "   " + str(probVal),
                (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 255), 1)
    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break