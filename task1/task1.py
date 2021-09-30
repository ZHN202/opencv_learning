import cv2 as cv
import numpy as np

img = cv.imread('test.png')

# 将图片大小改为1920*1080h,s,v = cv.split(hsvimg)
img = cv.resize(img, dsize=(1920, 1080), fx=1, fy=1, interpolation=cv.INTER_NEAREST)
# hsv图像
hsvimg = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# mpl.showimages(hsvimg,['hsvimg'])
# cv.imshow("hsv",hsvimg)
h, s, v = cv.split(hsvimg)
lower_y = np.array([15, 43, 46])
upper_y = np.array([65, 255, 255])
mask = cv.inRange(hsvimg, lower_y, upper_y)
# res = cv.bitwise_and(img,img,mask=mask)
cv.imshow('mask', mask)

# 获取灰度图
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 降噪
blurimg = cv.GaussianBlur(grayimg, (3, 3), 0)
cv.imshow('blurimg', blurimg)

# 二值化
ret, binimg = cv.threshold(blurimg, 95, 255, cv.THRESH_BINARY)
cv.imshow('binimg', binimg)

# 腐蚀
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
erodimg = cv.erode(binimg, kernel, iterations=1)
# cv.imshow('erodimg',erodimg)

# 获取轮廓
gradx = cv.Sobel(binimg, cv.CV_64F, 1, 0)
grady = cv.Sobel(binimg, cv.CV_64F, 0, 1)
gradx, grady = cv.convertScaleAbs(gradx), cv.convertScaleAbs(grady)
grad = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
edge = cv.Canny(grad, 50, 100)

# 霍夫直线检测
lines = cv.HoughLinesP(mask, 1, np.pi / 180, 127, minLineLength=100, maxLineGap=1)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
cv.imshow('img', img)
cv.waitKey(0)
