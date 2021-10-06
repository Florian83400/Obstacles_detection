# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 10:25:44 2021

@author: ferna
"""



########### Obstacle Detection ###########

########### Importing Useful Libraries ###########
import cv2
import numpy as np
import argparse

#### Connexion to the Camera ####
camera_index = 0
cap = cv2.VideoCapture(camera_index)

#### Reshape Video frame ####
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

#### Definition of Threshold Value ####
lowThreshold = 50
highThreshold = lowThreshold * 2




####### MAIN LOOP of CODE #######

while (True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # convert to grayscale
        
    # Gaussian Blurring
    blur = cv2.GaussianBlur(gray, (5,5), 0)   # blur the image
    
    
    ### Use of Canny Detector ###               
    img = cv2.Canny(blur,lowThreshold,highThreshold, (5,5), L2gradient = True)
    cv2.imshow('Canny Detector',img)
    
   # Finding contours for the Canny edge detector image
    img2,contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    # create an empty black images
    drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    drawing2 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    # Draw Contours on drawing image
    for i in range(len(contours)):
            color_contours = (0,255,0) # green - color for contours
            # draw ith contour
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy,cv2.LINE_AA)

    # create hull array for convex hull points
    hull =[]
        
        # calculate points for each contour
    for i in range(len(contours)):
        color = (255, 0, 0)
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i],False))
      
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)
        
    
    # Print final image
    cv2.imshow('Frame', frame)
    cv2.imshow('Image with contours',drawing)
    #cv2.imshow('Image with closed contours',drawing2)
    
    # Stop to broadcast
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




######## CODE UTILE ########

#    # Opening function as a filter
#    kernel = np.ones((3,3), np.uint8)
#    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN,kernel)

 # Find maximum area contour
#    cnt , max_index = maxContour(contours)    
#    cv2.drawContours(drawing2, [cnt],2, (0,255,0), -1)
    
    
#    # Define closed contours on image
#    closed_contours = []
#    
#    for cnt in contours:
#        if cv2.isContourConvex(cnt) == True:
#            closed_contours.append(cnt)
#        else:
#            pass
#        print('closed_contours', closed_contours) # print list of closed contours
#    
#    # Draw closed contours on drawing image
#    for i in range(len(closed_contours)):
#        color_closed_contours = (255, 0, 0) # blue - color for closed contours
#        #cv2.drawContours(drawing, [closed_contours], i, color_closed_contours, 1, 8,hierarchy, cv2.LINE_AA)

## Determine largest contour in the image
#def maxContour(contours):
#    cnt_list = np.zeros(len(contours))
#    for i in range(0, len(contours)):
#        cnt_list[i] = cv2.contourArea(contours[i])
#    
#    max_value = np.amax(cnt_list)
#    max_index = np.argmax(cnt_list)
#    cnt = contours[max_index]
#    
#    return cnt, max_index

#def find_largest_contour(contours):
#    largest_contour = max(contours, key = cv2.contourArea)
#    return largest_contour

#    hull =[]
#        
#    # calculate points for each contour
#    for i in range(len(contours)):
#        # creating convex hull object for each contour
#        hull.append(cv2.convexHull(contours[i],False))
#  
#    # draw ith convex hull object
#        cv2.drawContours(drawing, hull, i, color, 1, 8)

#    #Take only biggest contour basing on area
#    calcarea = 0.0
#    unicocnt = list()
#    for i in range (0, len(contours)):
#        area = cv2.contourArea(contours[i])
#        #print ('area', area)
#        if area > 90:
#            if calcarea < area:
#                calcarea = area
#                unicocnt = contours[i]
#        #print('calcarea', unicocnt)

    #cv2.drawContours(drawLargestContours, largest_contour, color_contours,1,8,hierarchy,cv2.LINE_AA)

#stereo = cv2.StereoBM_create(numDisparities = 96 , blockSize = 15)

#    disparity = stereo.compute(imgL, imgR)
#    cv2.imshow('disparity', disparity)

#    # Apply Sobelx in high output datatype 'float32'
#    #  and then converting back to 8-bit to prevent overflow
#    sobelx_64 = cv2.Sobel(blur, cv2.CV_32F,1,0,ksize = 3)
#    absx_64 = np.absolute(sobelx_64)
#    sobelx_8u1 = absx_64/absx_64.max()*255
#    sobelx_8u = np.uint8(sobelx_8u1)
#    
#    # Similarly for Sobely
#    sobely_64 = cv2.Sobel(blur, cv2.CV_32F,0,1,ksize = 3)
#    absy_64 = np.absolute(sobely_64)
#    sobely_8u1 = absy_64/absy_64.max()*255
#    sobely_8u = np.uint8(sobely_8u1)
#    
#    # From gradient calculate the magnitude and changing
#    # it to 8-bit
#    mag = np.hypot(sobelx_8u, sobely_8u)
#    mag = mag/mag.max()*255
#    mag = np.uint8(mag)
#    
#    # Find the direction and change it to degree
#    theta = np.arctan2(sobely_64, sobelx_64)
#    angle = np.rad2deg(theta)
#    
#    
#    
#    # Find the neighbouring pixels (b,c) in the rounded gradient direction
#    # and then apply non-max suppression
##    M, N = mag.shape
##    Non_max = np.zeros((M,N), dtype = np.uint8)
##    
##    for i in range(1, M-1):
##        for j in range(1,N-1):
##            # Horizontal 0
##            if (0 <= angle[i,j]<22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle [i,j] < -157.5):
##                b = mag[i, j+1]
##                c = mag[i, j-1]
##            # Diagonal 45
##            elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] <-112.5):
##                b = mag[i+1,j+1]
##                c = mag[i-1, j-1]
##            # Vertical 90
##            elif (67.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
##                b = mag[i+1, j]
##                c = mag[i-1, j]
##            # Diagonal 135
##            elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
##                b = mag[i+1, j-1]
##                c = mag[i-1, j+1]
##                
##            # Non-max Suppression
##            if (mag[i,j] >= b) and (mag[i,j] >= c):
##                Non_max[i,j] = mag[i,j]
##            else:
##                Non_max[i,j] = 0
#            
#  
#    # Set high and low threshold
#    highThreshold = 21
#    lowThreshold = 15
#    
#    M, N = angle.shape
#    out = np.zeros((M,N), dtype = np.uint8)
#    
#    # If edge intensity is greater than 'High' it is a sure-edge
#    # below 'low' threshold, it is a sure non-edge
#    strong_i, strong_j = np.where(angle >= highThreshold)
#    zeros_i, zeros_j = np.where(angle < lowThreshold)
#    
#    # Weak edges
#    weak_i, weak_j = np.where((angle <= highThreshold) & (angle >= lowThreshold))
#    
#    #Set same intensity value for all edge pixels
#    out[strong_i, strong_j] = 255
#    out[zeros_i, zeros_j] = 0
#    out[weak_i, weak_j] = 75
#    
#    M, N = out.shape
#    for i in range(1,M-1):
#        for j in range(1, N-1):
#            if (out[i,j] == 75):
#                if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
#                    out[i,j] = 255
#                else:
#                    out[i,j] = 0