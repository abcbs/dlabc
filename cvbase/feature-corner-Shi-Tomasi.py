#coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
path='D:\\DevN\\sample-data\\opencv-image\\'
filename=path+"blox.jpg"

img = cv2.imread(filename)

#检查角点特征
# Harris 角点检测的结果是一个由角点分数构成的灰度图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray,60,0.01,10)
# 返回的结果是[[ 311., 250.]] 两层括号的数组。
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()