import numpy as np
import cv2 as cv
from practise import mpl

# 读取图片
img = cv.imread('test2.jpg')

# 转化为灰度图
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 高斯模糊
gussimg = cv.GaussianBlur(grayimg, (3, 3), 0)

# 二值化
ret, binimg = cv.threshold(gussimg, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# 寻找轮廓
contours1, hierarchy = cv.findContours(binimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 多边形逼近
approx = cv.approxPolyDP(contours1[0], 7, True)

# 透视变换
imgHeight = img.shape[0]
imgWidth = img.shape[1]
imgmat = np.float32([[0, 0], [0, imgHeight - 1], [imgWidth - 1, 0], [imgWidth - 1, imgHeight - 1]])
contmat = np.float32([tuple(approx[0][0]), tuple(approx[1][0]), tuple(approx[3][0]), tuple(approx[2][0])])
mat = cv.getPerspectiveTransform(contmat, imgmat)
new = cv.warpPerspective(img, mat, (imgWidth, imgHeight))

# 转化为灰度图
gray = cv.cvtColor(new, cv.COLOR_BGR2GRAY)

# 二值化
ret, binimg = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# 闭运算--先膨胀再腐蚀
element = cv.getStructuringElement(cv.MORPH_RECT, (18, 18))
close = cv.erode(binimg, element)
close = cv.dilate(close, element)
close = cv.morphologyEx(close, cv.MORPH_CLOSE, element)

# 再次二值化
ret, binimg = cv.threshold(close, 127, 255, cv.THRESH_BINARY_INV)

# 找轮廓
contours2, hierarchy = cv.findContours(binimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(new, contours2, -1, (0, 255, 0), 3)

# 计算面积
phonearea = cv.contourArea(contours2[0])
area = 21 * 29.7 * (phonearea / (imgWidth * imgHeight))
cv.putText(new,str(area),(int(imgWidth/3),int(imgHeight/7)),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=3)
mpl.showimages(new,['res'])