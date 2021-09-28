import cv2
import numpy as np
import cv2 as cv

vc = cv2.VideoCapture("D:/cv2/test.mp4")
img = cv2.imread("1.png")

person = img[0:200,0:200]
cv2.imshow('2', person)
cv2.imshow('1',img)

if vc.isOpened():
    open, fram = vc.read()
else:
    open = False

while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
        if cv2.waitKey(100) & 0xFF == 27:
            break

vc.release()
cv2.destroyAllWindows()

