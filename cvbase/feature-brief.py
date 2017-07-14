#coding=utf-8
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

path='D:\\DevN\\sample-data\\opencv-image\\'
filename=path+"messi5.jpg"

img = cv2.imread(filename)

# Initiate STAR detector
star = cv2.FlannBasedMatcher_create()

# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create("BRIEF")

# find the keypoints with STAR
kp = star.detect(img,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

print(brief.getInt('bytes'))
print (des.shape)