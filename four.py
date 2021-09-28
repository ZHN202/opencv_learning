import cv2 as cv
import mpl
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("test.png")

grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, img4 = cv.threshold(grayimg,80,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
img2 = cv.adaptiveThreshold(grayimg,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
cv.imshow("img2",img4)
img3 = cv.blur(img2,(3,3),0)
cv.imshow("img3",img3)
reimg = cv.resize(img, dsize=(1920, 1080), fx=1, fy=1, interpolation=cv.INTER_NEAREST)

ret, binaryimg = cv.threshold(grayimg, 90, 255, cv.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)

cannyimg = cv.Canny(grayimg, 50, 100)
cv.imwrite('c.png',cannyimg)
titles = ["origin", "grayimg", "reimg", "binaryimg", "gradent"]
imgs = [img, grayimg, reimg, binaryimg, cannyimg]
mpl.showimages(imgs, titles=titles, num_cols=2, axis='on')
