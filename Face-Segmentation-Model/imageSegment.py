# PERIVITTA RAJENDRAN 1171101579
# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

# input_dir = 'add_dataset/add_test'
# output_dir = 'add_dataset/output'
input_dir = 'dataset/test'
output_dir = 'dataset/output'
# # You are allowed to import other Python packages above
# ################################################################################################
def segmentImage(img):

    img_hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # converted BGR to HSV first
    img_hsv =  cv2.medianBlur(img_hsv1,3) # the images has better output when its blurred
    # kernel = np.ones((3,3),np.uint8) # kernel 3,3 or kernel 5,5 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,5))
    # Easier to refer if the code is working or not 
    print("Segmenting...........")   
    # print("BKG"*6)
    # The '0' refers to intensity of background. 
    lower_bg = np.array([0 , 0, 178], dtype = "uint8")
    upper_bg = np.array([149, 15, 255], dtype = "uint8")
    mask = cv2.inRange(img_hsv, lower_bg, upper_bg)

    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 1)
    ret,th = cv2.threshold(close,40,0,cv2.THRESH_BINARY)
################################################################################################
    # print("HAIRRRRR"*6)
    lower_hair = np.array([0 , 0, 0], dtype = "uint8")
    upper_hair = np.array([180, 255, 20], dtype = "uint8")
    mask1 = cv2.inRange(img_hsv, lower_hair, upper_hair)

    opening1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations = 1)
    ret1,th1 = cv2.threshold(opening1,40,1,cv2.THRESH_BINARY)
################################################################################################
    # print("MOUTHHHHH"*6)
    # image = cv2.medianBlur(img, 3)  ### I've' already blurred in the beginning
    lower_mouth = np.array([0 , 82, 0],dtype = "uint8")
    upper_mouth = np.array([3, 255, 255],dtype = "uint8")
    mask2 = cv2.inRange(img_hsv, lower_mouth, upper_mouth)
    image2 = cv2.medianBlur(mask2,9)
    
    opening2 = cv2.morphologyEx(image2, cv2.MORPH_OPEN, kernel, iterations = 1)
    ret2,th2 = cv2.threshold(opening2,40,2,cv2.THRESH_BINARY)
################################################################################################   
    # print("EYESSSSSS"*6)
    # lower_eyes = np.array([70,18,71],dtype = "uint8") ## second version has better accuracy
    # upper_eyes = np.array([90,38,151],dtype = "uint8")  
    lower_eyes = np.array([0,0,125],dtype = "uint8")
    upper_eyes = np.array([145,48,165],dtype = "uint8")  
    mask3 = cv2.inRange(img_hsv, lower_eyes, upper_eyes)
    # image3 = cv2.medianBlur(mask3,3)

    dilation3 = cv2.dilate(mask3, kernel, iterations =1)
    ret3,th3 = cv2.threshold(dilation3,70,2,cv2.THRESH_BINARY)
    
    lower_eyes1 = np.array([0,0,0],dtype = "uint8")
    upper_eyes1 = np.array([179,50,165],dtype = "uint8")  
    mask33 = cv2.inRange(img_hsv, lower_eyes1, upper_eyes1)
    # image3 = cv2.medianBlur(mask3,3)
    dilation33 = cv2.dilate(mask33, kernel, iterations =1)
    ret33,th33 = cv2.threshold(dilation33,70,2,cv2.THRESH_BINARY)  
            
    lower_eye4 = np.array([0 , 0, 92], dtype = "uint8")
    upper_eye4 = np.array([177, 43, 181], dtype = "uint8")
    mask43 = cv2.inRange(img_hsv, lower_eye4, upper_eye4)#masking hsv image
    
    opening43 = cv2.morphologyEx(mask43, cv2.MORPH_OPEN, kernel, iterations = 1)
    ret43,th43 = cv2.threshold(opening43,60,3,cv2.THRESH_BINARY)   
################################################################################################        
    ###### HAIRRR THIS ONE 
    lower_hair1 = np.array([117,10,163],dtype = "uint8")
    upper_hair1 = np.array([121,255,255],dtype = "uint8")  
    mask11 = cv2.inRange(img_hsv, lower_hair1, upper_hair1)
    # image3 = cv2.medianBlur(mask3,3)
    dilation11 = cv2.dilate(mask11, kernel, iterations =1)
    ret11,th11 = cv2.threshold(dilation11,70,2,cv2.THRESH_BINARY)
################################################################################################    
    # print("NOSEEEEE"*6)
    nose_lower = np.array([0, 201, 28], dtype = "uint8")  #nose
    nose_upper = np.array([18, 229, 73], dtype = "uint8")
    # nose_lower = np.array([5, 50, 102], dtype = "uint8")  #nose
    # nose_upper = np.array([179, 190, 255], dtype = "uint8")
    mask4 = cv2.inRange(img_hsv, nose_lower, nose_upper)
    close4 = cv2.morphologyEx(mask4, cv2.MORPH_CLOSE, kernel, iterations = 1)
    ret4,th4 = cv2.threshold(close4,40,4,cv2.THRESH_BINARY)

    # nose_lower1 = np.array([0, 0, 0], dtype = "uint8")  #nose
    # nose_upper1 = np.array([20, 230, 100], dtype = "uint8")
    nose_lower1 = np.array([5, 50, 102], dtype = "uint8")  #nose
    nose_upper1 = np.array([179, 190, 255], dtype = "uint8")
    mask44 = cv2.inRange(img_hsv, nose_lower1, nose_upper1)
    close44 = cv2.morphologyEx(mask44, cv2.MORPH_CLOSE, kernel, iterations = 1)
    ret44,th44 = cv2.threshold(close44,40,4,cv2.THRESH_BINARY)
################################################################################################
    # print("SKIN"*6)
    lower_skin = np.array([0, 48, 80], dtype = "uint8")
    upper_skin = np.array([20, 255, 255], dtype = "uint8")
    skinMask = cv2.inRange(img_hsv, lower_skin, upper_skin)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel2, iterations = 1)
    skinMask = cv2.dilate(skinMask, kernel2, iterations = 1)
    skinMask = cv2.GaussianBlur(skinMask, (1, 1), 0)
    ret5,th5 = cv2.threshold(skinMask,0,5,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
################################################################################################ 
    masks = th + th1 + th2 + th3 + th4 + th5 + th11 +  th44 + th43

    outImg = np.clip(masks,0,255) 
    # END OF YOUR CODE
################################################################################################
    return outImg
### References [Color codes is in ipynb, if method 2 is used]
# 01 - background 
# 02 - Hair and brows
# 03 - mouth
# 04 - eyes
# 05 - nose
# 06 - skin
################################################################################################