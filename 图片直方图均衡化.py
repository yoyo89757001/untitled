import cv2
import numpy as np


#灰度图
# img = cv2.imread("timg3.jpeg",1)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("fdsf",gray)
# dd = cv2.equalizeHist(gray)
# cv2.imshow('ddd',dd)


#彩色图
# img = cv2.imread("timg3.jpeg",1)
# cv2.imshow("fdsf",img)
# (b,g,r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# result = cv2.merge((bH,gH,rH))
# cv2.imshow('ddd',result)

#YUV
img = cv2.imread("timg3.jpeg",1)
cv2.imshow("fdsf",img)
imageYUV = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
chnnelYUV = cv2.split(imageYUV)
chnnelYUV[0] = cv2.equalizeHist(chnnelYUV[0])
chnnels = cv2.merge(chnnelYUV)
result = cv2.cvtColor(chnnels,cv2.COLOR_YCrCb2BGR)

cv2.imshow('ddd',result)

cv2.waitKey(0)