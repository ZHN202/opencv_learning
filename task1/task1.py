import cv2 as cv
import numpy as np

img = cv.imread('test.png')

# 将图片大小改为1920*1080h,s,v = cv.split(hsvimg)
img = cv.resize(img, dsize=(1920, 1080), fx=1, fy=1, interpolation=cv.INTER_NEAREST)

# hsv图像
hsvimg = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower_y = np.array([20, 43, 46])
upper_y = np.array([34, 255, 220])
mask = cv.inRange(hsvimg, lower_y, upper_y)

# 霍夫直线检测
lines = cv.HoughLinesP(mask, 1, np.pi / 180, 127, minLineLength=500, maxLineGap=1)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
cv.imshow('img', img)
cv.waitKey(0)
