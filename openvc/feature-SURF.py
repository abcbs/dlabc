#coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:\\DevN\\sample-data\\images\\fly.png',3)
# img1 = cv2.imread('D:\\DevN\\sample-data\\images\\football\\messi5.jpg',3)
# img2 = cv2.imread('D:\\DevN\\sample-data\\images\\football\\messi5-3.jpg',3)

# surf = cv2.SURF(400)
surf = cv2.xfeatures2d.SURF_create(40000)

kp, des = surf.detectAndCompute(img,None)

print(len(kp))

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2),plt.show()

