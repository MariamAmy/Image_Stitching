# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 21:03:03 2022

@author: EL-Handasia
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

img1 = cv2.imread('E:/Miscellaneous/ROV/Digital Image Processing/building/building/building1.JPG')
img11 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('E:/Miscellaneous/ROV/Digital Image Processing/building/building/building2.JPG')
img22 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img3 = cv2.imread('E:/Miscellaneous/ROV/Digital Image Processing/building/building/building3.JPG')
img33 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

img4 = cv2.imread('E:/Miscellaneous/ROV/Digital Image Processing/building/building/building4.JPG')
img44 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

img5 = cv2.imread('E:/Miscellaneous/ROV/Digital Image Processing/building/building/building5.JPG')
img55 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
#keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img11,None)
kp2, des2 = sift.detectAndCompute(img22,None)
kp3, des3 = sift.detectAndCompute(img33,None)
kp4, des4 = sift.detectAndCompute(img44,None)
kp5, des5 = sift.detectAndCompute(img55,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, des3, des4, des5, k=2) 
