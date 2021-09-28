import cv2 as cv
import numpy as np
import mpl

yellowImg = cv.imread("task3-2.png")
cap = cv.VideoCapture("testvideo.mp4")
fps = cap.get(cv.CAP_PROP_FPS)
frameCount = cap.get(cv.CAP_PROP_FRAME_COUNT)
redImg = cv.imread("task3-3.png")

def detectTrafficlight(pic):
    pic_hsv = cv.cvtColor(pic, cv.COLOR_BGR2HSV)
    mpl.showimages(pic_hsv, ["hsv"])

mpl.showimages(cv.cvtColor(yellowImg,cv.COLOR_BGR2HSV),["red"])
mpl.showimages(cv.cvtColor(redImg,cv.COLOR_BGR2HSV), ['yell'])
while (cap.isOpened()):
    ret, frame = cap.read()
    detectTrafficlight(frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
