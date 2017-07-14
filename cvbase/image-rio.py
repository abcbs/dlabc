#coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
path='D:\\DevN\\sample-data\\opencv-image\\'
filename=path+"messi5.jpg"

img = cv2.imread(filename)

ball = img[280:340, 330:390]

cv2.imshow('dst',ball)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()