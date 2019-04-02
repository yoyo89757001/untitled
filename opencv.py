import cv2


im = cv2.imread('test.png',1)
#cv2.imshow('qwqw',im)
#cv2.imwrite('sdf.png',im,[cv2.IMWRITE_PNG_COMPRESSION,9])
#cv2.imwrite('sdf.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,90])
#opencv中读取出来的像素以bgr顺序格式取出。一般其它软件是 rgb顺序
(b,g,r) = im[100,100]

im[:,100] = (255,0,0)

cv2.imshow('www',im)
cv2.waitKey(0)
print(b,g,r)