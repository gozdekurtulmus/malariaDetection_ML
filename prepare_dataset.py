# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:31:43 2021

"""

import cv2
import glob

#Taking our dataset

label = "Parasitized" #  "Uninfected"
dirList = glob.glob("cell_images/" + label + "/*.png")
file = open("csv/dataset.csv","a")

#Iterate over every image
for img_path in dirList:
    im = cv2.imread(img_path)
    im_blurred = cv2.GaussianBlur(im,(5,5),2)  # Smoothen the image
 
    im_gray = cv2.cvtColor(im_blurred, cv2.COLOR_BGR2GRAY) #Convert to grayscale
    
    ret,thresh = cv2.threshold(im_gray, 127,255,0)
    contours,_ = cv2.findContours(thresh,1,2) #Contour Detection
    
    file.write(label)
    file.write(",")
    
    for i in range(5):
        try:
            area = cv2.contourArea(contours[i])
            file.write(str(area))
        except:
            file.write("0")
        if (i < 4):
            file.write(",")
        
    file.write("\n")
file.close()