import cv2 as cv
import numpy as np


cap = cv.VideoCapture("testvideo.mp4")
fps = cap.get(cv.CAP_PROP_FPS)
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter("result.avi",fourcc,fps,(height,width))




def detect(pic):
    picHSV = cv.cvtColor(pic, cv.COLOR_BGR2HSV)
    # 获取aruco码部分的掩膜
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    mask = cv.inRange(picHSV, lower_black, upper_black)
    # 提取aruco码
    contours, h = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    approx = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if 0.8<w/h<=1 and 2500<w*h<5000:
            approx.append(cv.approxPolyDP(contour, 7, True))
            cv.drawContours(pic, approx, -1, (0, 0, 255), 3)
    if len(approx) == 2:
    # 提取感兴趣区域
        Roi = pic[0:approx[0][0][0][1],
          int((approx[0][1][0][0] + approx[0][0][0][0]) / 2):
          int((approx[1][0][0][0] + approx[1][1][0][0]) / 2)]
        gray = cv.cvtColor(Roi, cv.COLOR_BGR2GRAY)
        ret, bin = cv.threshold(gray, 240, 255, cv.THRESH_BINARY)
        x = np.shape(bin)[1]
        y = np.shape(bin)[0]
        for i in range(x):
            flag = False
            for j in range(y):
                if bin[j][i] == 255:
                    flag = True
                    break
            if flag is True:
                break
        light = ''
        if 0.3 < j / y < 0.6:
            light = 'yellow'
        elif 0 < j / y <= 0.3:
            light = 'red'
        elif 0.6 <= j / y < 1:
            light = 'green'
        cv.putText(pic,light,(j,i),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(0,255,0))
    return pic


i = 0
ret,frame = cap.read()
while ret:
    i = i + 1
    if i >= 10:
        res = detect(frame)
        cv.imshow('res',res)
        cv.waitKey(1000 // int(fps))
        out.write(res)
    ret, frame = cap.read()
cv.destroyAllWindows()
cap.release()