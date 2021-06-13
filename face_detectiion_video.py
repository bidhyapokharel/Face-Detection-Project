import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default')
cap = cv2.VideoCapture('test.mp4')

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    # Display the output
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
