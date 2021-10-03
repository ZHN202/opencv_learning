from practise import mpl
import cv2 as cv
import numpy as np

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
ret, yellowLightBin = cv.threshold(yellowLightGray, 40, 255, cv.THRESH_BINARY)
ret, greenLightBin = cv.threshold(greenLightGray, 40, 255, cv.THRESH_BINARY)
ret, redLightBin = cv.threshold(redLightGray, 40, 255, cv.THRESH_BINARY)
imgs = [greenLightBin, yellowLightBin, redLightBin]
titles = ["gb", "yb", "rb"]
mpl.showimages(imgs, titles)
mpl.showimages(yellowLightBin, ['y'])

# HSV
ylHSV = cv.cvtColor(yellowLight, cv.COLOR_BGR2HSV)
glHSV = cv.cvtColor(greenLight, cv.COLOR_BGR2HSV)
redHSV = cv.cvtColor(redLight, cv.COLOR_BGR2HSV)
imgs = [glHSV, ylHSV, redHSV]
titles = ["y", "g", "r"]
mpl.showimages(imgs, titles)

# 通道分离
# H, S, V = cv.split(ylHSV)
lower_blue = np.array([0, 0, 0])
upper_blue = np.array([180, 255, 60])
mask = cv.inRange(ylHSV, lower_blue, upper_blue)
mpl.showimages(mask, ['mask'])

# 提取aruco码
knel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
close = cv.morphologyEx(mask, cv.MORPH_OPEN, knel)
close = cv.morphologyEx(close, cv.MORPH_OPEN, knel)
mpl.showimages(close, ['close'])
conts, h = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

approx = []

for cont in conts:
    x,y,w,h = cv.boundingRect(cont)
    if 0.8<w/h<=1 and 25000<w*h<50000:
        approx.append(cv.approxPolyDP(cont, 7, True))
        cv.drawContours(yellowLight, approx, -1, (0, 0, 255), 3)
mpl.showimages(yellowLight, ['newimg'])

# 提取红绿灯
lower_red = np.array([0,43,46])
upper_red = np.array([10,255,255])
lower_yellow = np.array([11,43,46])
upper_yellow = np.array([25,255,255])
lower_green = np.array([35,43,46])
upper_green = np.array([77,255,255])
mask_red = cv.inRange(ylHSV, lower_red, upper_red)
mask_yellow = cv.inRange(ylHSV, lower_yellow, upper_yellow)
mask_green = cv.inRange(ylHSV, lower_green, upper_green)
masks = [mask_red,mask_green,mask_yellow]
titles = ['red','green','yellow']
mpl.showimages(masks,titles)
masks = mask_red+mask_green+mask_yellow
mpl.showimages(masks,['2'])
close = cv.morphologyEx(mask_yellow, cv.MORPH_OPEN, knel)
close = cv.morphologyEx(close, cv.MORPH_OPEN, knel)
mpl.showimages(close,['1'])


ROI = np.zeros(yellowLight.shape, np.uint8)

maxRoi = yellowLight[0:approx[0][0][0][1],int((approx[0][1][0][0]+approx[0][0][0][0])/2):int((approx[1][0][0][0]+approx[1][1][0][0])/2)]
mpl.showimages(maxRoi,['3'])

HSVRoi = cv.cvtColor(maxRoi,cv.COLOR_BGR2HSV)
mpl.showimages(HSVRoi,['4'])

H,S,V = cv.split(HSVRoi)

gray = cv.cvtColor(maxRoi,cv.COLOR_BGR2GRAY)
mpl.showimages(gray,['5'])

ret,bin  = cv.threshold(gray,240,255,cv.THRESH_BINARY)
mpl.showimages(bin,['6'])
print(np.shape(bin))
x = np.shape(bin)[1]
y = np.shape(bin)[0]
for i in range(x):
    flag = False
    for j in range(y):
        if bin[j][i]==255:
            flag=True
            break
    if flag is True:
        break
if 0.3<j/y<0.6:
    light = 'yellow'
    print(light)