#coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
path='D:\\DevN\\sample-data\\opencv-image\\'
filename=path+"vases.jpg"

img = cv2.imread(filename)

#转为灰度格式
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#尺度不变特征变化
sift = cv2.xfeatures2d.SURF_create(6000)

keypoints, descriptor = sift.detectAndCompute(gray,None)

img = cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51,163,236))


cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()