import cv2
import numpy as np

im = cv2.imread('timg3.jpeg',1)
#cv2.imshow('qwqw',im)
#cv2.imwrite('sdf.png',im,[cv2.IMWRITE_PNG_COMPRESSION,9])
#cv2.imwrite('sdf.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,90])
#opencv中读取出来的像素以bgr顺序格式取出。一般其它软件是 rgb顺序

#im[:,100] = (255,0,0)

imageInfo = im.shape

w = imageInfo[1]
h= imageInfo[0]
ss=imageInfo[2]

matRotate = cv2.getRotationMatrix2D((h*0.5,w*0.5),45,0.5)
dst = cv2.warpAffine(im,matRotate,(h,w))


#gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# #   灰度图
# dst2 = np.zeros((h,w,1),np.uint8)
# for i in range(0,h):
#     for j in range(0,w):
#         (b,g,r) = im[i,j]
#         b = int(b)
#         g = int(g)
#         r = int(r)
#        # gray = b*0.22+g*0.33+r*0.14
#         gray2 = (r+(g<<1)+b)>>2
#         dst2[i,j] = np.uint8(gray2)


#彩色反转
# dst2 = np.zeros((h,w,3),np.uint8)
# for i in range(0,h):
#     for j in range(0,w):
#         (b,g,r) = im[i,j]
#         dst2[i,j] = (255-b,255-g,255-r)


#马赛克
# for m in range(100,300):
#     for n in range(100,200):
#         if m%10 == 0 and n%10 == 0 :
#             for i in range(0,10):
#                 for j in range(0,10):
#                     (b,g,r) = im[m,n]
#                     im[i+m,j+n] = (b,g,r)



#毛玻璃
# dst3 = np.zeros((h,w,3),np.uint8)
# mm = 8
# for m in range(0,h-mm):
#     for n in range(0,w-mm):
#         index =int(np.random.random()*mm)
#         (b,g,r) = im[m+index,n+index]#当前位置加一个随机数，得到一个附近新的位置
#         dst3[m,n] = (b,g,r)


#边缘检测
# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# imgG = cv2.GaussianBlur(gray,(3,3),0) #高斯滤波
# ddst = cv2.Canny(imgG,50,50)# 边缘检测。 第二第三个参数是阈值(图片的卷积大于这个阈值，就说明是边缘点)


cv2.imshow("sss",ddst)
cv2.waitKey()