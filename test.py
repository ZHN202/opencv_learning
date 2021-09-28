import mpl
import cv2 as cv

yellowLight = cv.imread("task3-2.png")
greenLight = cv.imread("task3-1.png")
redLight = cv.imread("task3-3.png")
imgs = [greenLight, yellowLight, redLight]
titles = ["green", "yellow", "red"]
mpl.showimages(imgs, titles)

# 灰度图
yellowLightGray = cv.cvtColor(yellowLight, cv.COLOR_BGR2GRAY)
greenLightGray = cv.cvtColor(greenLight, cv.COLOR_BGR2GRAY)
redLightGray = cv.cvtColor(redLight, cv.COLOR_BGR2GRAY)
imgs = [greenLightGray, yellowLightGray, redLightGray]
titles = ["gg", "gy", "gr"]
mpl.showimages(imgs, titles)

# 二值化
ret, yellowLightBin = cv.threshold(yellowLightGray, 50, 255, cv.THRESH_BINARY)
ret, greenLightBin = cv.threshold(greenLightGray, 50, 255, cv.THRESH_BINARY)
ret, redLightBin = cv.threshold(redLightGray, 50, 255, cv.THRESH_BINARY)
imgs = [greenLightBin, yellowLightBin, redLightBin]
titles = ["gb", "yb", "rb"]
mpl.showimages(imgs, titles)
mpl.showimages(yellowLightBin,['y'])

# HSV
ylHSV = cv.cvtColor(yellowLight,cv.COLOR_BGR2HSV)
glHSV = cv.cvtColor(greenLight,cv.COLOR_BGR2HSV)
redHSV = cv.cvtColor(redLight,cv.COLOR_BGR2HSV)
imgs = [glHSV,ylHSV,redHSV]
titles = ["y","g","r"]
mpl.showimages(imgs,titles)