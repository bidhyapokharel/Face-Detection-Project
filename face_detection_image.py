import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default')
eye_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default')

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for x, y, w, h in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    roi_gray = gray[y:y+h, x:x+h]
    roi_color = img[y:y+h, x:x+h]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (x,y), (ex+ew, ey+eh), (0,255,0), 2)
        

cv2.imshow('Image', img)
cv2.waitKey()
# cv2.destroyAllWindows()