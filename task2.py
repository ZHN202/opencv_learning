import numpy as np
import cv2 as cv
import mpl

# 读取图片
img = cv.imread('test2.jpg')

# 转化为灰度图
grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
mpl.showimages(grayimg, ["grayimg"])

# 高斯模糊
gussimg = cv.GaussianBlur(grayimg, (3, 3), 0)
# mpl.showimages(gussimg, ["gussimg"])

# 二值化
ret, binimg = cv.threshold(gussimg, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
mpl.showimages(binimg, ["binimg"])

# 寻找轮廓
contours, hierarchy = cv.findContours(binimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contimg = cv.drawContours(img, contours, 0, (0, 0, 255), 1)
mpl.showimages(contimg, ["contimg"])
# 多边形逼近
approx = cv.approxPolyDP(contours[0], 7, True)

for i in range(4):
    point = tuple(approx[i][0])
    print(type(point[0]))
    img = cv.circle(img, point, 1, (0, 255, 0), 0)
mpl.showimages(img, ["img"])
# 透视变换
imgHeight = img.shape[0]
imgWidth = img.shape[1]
imgmat = np.float32([[0, 0], [0, imgHeight - 1], [imgWidth - 1, 0], [imgWidth - 1, imgHeight - 1]])
contmat = np.float32([tuple(approx[0][0]), tuple(approx[1][0]), tuple(approx[3][0]), tuple(approx[2][0])])
mat = cv.getPerspectiveTransform(contmat, imgmat)
new = cv.warpPerspective(img, mat, (imgWidth, imgHeight))
mpl.showimages(new, ["new"])

# 转化为灰度图
gray = cv.cvtColor(new, cv.COLOR_BGR2GRAY)

# 二值化
ret, binimg = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
mpl.showimages(binimg, ["bin"])

# 闭运算--先膨胀再腐蚀
element = cv.getStructuringElement(cv.MORPH_RECT, (18, 18))
close = cv.erode(binimg, element)
close = cv.dilate(close, element)
close = cv.morphologyEx(close, cv.MORPH_CLOSE, element)
mpl.showimages(close, ["close"])
# 再次二值化
ret, binimg = cv.threshold(close, 127, 255, cv.THRESH_BINARY_INV)
mpl.showimages(binimg, ["binimg"])

# 找轮廓
contours, hierarchy = cv.findContours(binimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(new, contours, -1, (0, 255, 0), 3)
mpl.showimages(new, ['newimg'])

# 计算面积
phonearea = cv.contourArea(contours[0])
print(phonearea)
print(21 * 29.7 * (phonearea / (imgWidth * imgHeight)))
