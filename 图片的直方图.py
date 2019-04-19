import cv2
import numpy as np

def ImageHist(image,type):
    color = (255,255,255)
    windowsName = 'Gray'
    if type == 31:
        color = (255,0,0)
        windowsName = 'B Hist'
    elif type ==32:
        color = (0, 255, 0)
        windowsName = 'G Hist'
    elif type ==33:
        color = (0, 0, 255)
        windowsName = 'R Hist'
    hist = cv2.calcHist([image],[0],None,[256],[0.0,255.0])
    minV,maxV,minL,maxL = cv2.minMaxLoc(hist)
    histImage = np.zeros([256,256,3],np.uint8)
    for h in range(256):
        intenNormal = int(hist[h]*256/maxV)
        cv2.line(histImage,(h,256),(h,256-intenNormal),color)

    cv2.imshow(windowsName,histImage)
    return histImage


img = cv2.imread("timg3.jpeg",1)
channels = cv2.split(img)# 拆分成B G R
for i in range(0,3):
    ImageHist(channels[i],31+i)

cv2.waitKey(0)
