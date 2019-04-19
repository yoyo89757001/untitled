import cv2
import numpy as np


newImageInfo = (500,500,3)
dst = np.zeros((newImageInfo),np.uint8)
#cv2.line(dst,(100,100),(400,400),(0,0,255),80,cv2.LINE_AA)

#cv2.ellipse(dst,(250,250),(100,100),0,0,100,(0,0,255),-1)

points = np.array([[10,5],[40,10],[70,20],[20,30]], np.int32)
points = points.reshape((-1, 1, 2))
cv2.polylines(dst, [points], True, (0,255,255))
font= cv2.FONT_HERSHEY_COMPLEX_SMALL
cv2.putText(dst,'t阿斯顿啊 ',(10,300),font,1,(0,0,222),1,cv2.LINE_AA)

cv2.calcHist()


cv2.imshow('sda',dst)
cv2.waitKey(0)
