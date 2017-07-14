#coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
path='D:\\DevN\\sample-data\\opencv-image\\'
filename=path+"box-macth.png"
img1 = cv2.imread(filename,0)

filename=path+"box-macth.png"
img2 = cv2.imread(filename,0)


# Initiate SIFT detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# resultImage = cv2.drawMatchesKnn(queryImage,kp1,trainingImage,kp2,matches,None,**drawParams)
imgOut=cv2.drawMatches(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2,matches1to2=matches[:40],outImg=None,flags=2)
plt.imshow(imgOut,),plt.show()