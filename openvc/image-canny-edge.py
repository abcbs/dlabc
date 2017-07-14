#coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:\\DevN\\sample-data\\images\\football\\messi5.jpg',3)
img1 = cv2.imread('D:\\DevN\\sample-data\\images\\football\\messi5.jpg',3)
img2 = cv2.imread('D:\\DevN\\sample-data\\images\\football\\messi5-3.jpg',3)

edges = cv2.Canny(img,100,250)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()