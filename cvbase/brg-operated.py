#coding=utf-8
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
# path='D:\\DevN\\sample-data\\opencv-image\\'
# filename=path+"blox.jpg"

# img = cv2.imread(filename)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)
#
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
#
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
img=np.zeros((3,3) ,dtype=np.uint8)
print(img.shape)

img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
print(img.shape)

#
radomArray=bytearray(os.urandom(120000))
flatNumpyArray=np.array(radomArray)
#gray
grayArray=flatNumpyArray.reshape(300,400)
#
bgrImage=flatNumpyArray.reshape(200,200,3)


b=bgrImage[:,:,0]
# bgrImage[:,:,0]=200
# bgrImage[:,:,1]=200
# bgrImage[:,:,2]=255
print(b)
bgrImage[10:100, 30:190,0:2]=23
cv2.imshow('dst',bgrImage)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()