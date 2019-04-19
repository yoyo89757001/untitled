import cv2
import numpy as np

#双边滤波 美白
# img =cv2.imread('timg3.jpeg',1)
# cv2.imshow('das',img)
# dst = cv2.bilateralFilter(img,50,60,60)
# cv2.imshow('dss',dst)
# cv2.waitKey(0)


#高斯均值滤波 美白
img =cv2.imread('timg3.jpeg',1)
cv2.imshow('das',img)
dst = cv2.GaussianBlur(img,(5,5),1.5)
cv2.imshow('dss',dst)
cv2.waitKey(0)