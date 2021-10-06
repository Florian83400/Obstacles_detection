# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:29:06 2021

@author: ferna
"""

### METHOD 1 #####


############ Background Substractor ###########
#
############ Importing Useful Libraries ###########
import cv2
import numpy as np
import argparse

#### Connexion to the Camera ####
camera_index = 0
cap = cv2.VideoCapture(camera_index)

#### Reshape Video frame ####
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

#setup args
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",help="pathto the video file")
ap.add_argument("-a","--min-area",type=int,default=2000, help="minimum area size")
args = vars(ap.parse_args())

#### Subtractor ####
subtractor = cv2.createBackgroundSubtractorMOG2(history = 1, varThreshold = 125, detectShadows = False)


####### MAIN LOOP of CODE #######
while True:
    _, frame = cap.read()
    
    ## Apply the subtractor ##
    
    sub = subtractor.apply(frame)
    
    img2,contours, hierarchy = cv2.findContours(sub, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    for c in contours:
        
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y),(x + w, y + h),(0, 0, 255), 2)
    
#    # Shape of image
#    rows, cols = sub.shape
    
#    x = []
#    y = []
#    for i in range(rows):
#        for j in range(cols):
#            k = sub[i,j]
#            
#            if k == 255:
#                x = i
#                y = j
            
#    cv2.rectangle(sub,(x,y), (x,y), (0,0,255), 2)
                
    
    ## Show results ##
    cv2.imshow('frame', frame)
    cv2.imshow('substractor', sub)
    
    # Stop to broadcast
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



#### METHOD 2 ####

#cap = cv2.VideoCapture(0)
#
###### Reshape Video frame ####
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
#
## Read the frame 1
#ret, frame1 = cap.read()
##cv2.imshow('Frame 1', frame1)
#
#while cap.isOpened():
#    # read the frame 2
#    ret, frame2 = cap.read()
#    cv2.imshow('Frame', frame2)
#    
#    # Extract the foreground mask
#    fgMask = cv2.absdiff(frame1, frame2)
#    #cv2.imshow('Foreground Mask', fgMask)
#    
#    # Apply the threshold for increasing white foreground
#    _, thresh = cv2.threshold(fgMask, 50, 250, cv2.THRESH_BINARY)
#    cv2.imshow('Foreground Mask', thresh)
#    
#    # assign frame2 to frame1 to continue the iteration untill all frames are read
#    frame1 = frame2
#    
#    # Wait for any key to be presed
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
## release video capture
#cap.release()
#cv2.destroyAllWindows()

### METHOD 3###

#cap = cv2.VideoCapture(0)
#
###### Reshape Video frame ####
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
#
## initialize Background subtractor for KNN
#BS_KNN = cv2.createBackgroundSubtractorKNN( history = 0, dist2Threshold = 150 )
#
#
#while cap.isOpened():
#    ret, frame = cap.read()
#    
#    # Extract the KNN-method of foreground Mask
#    knn_FGMask = BS_KNN.apply(frame)
#    cv2.imshow('KNN - Method : Foreground Mask', knn_FGMask)
#    
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#
##release video capture
#cv2.destroyAllWindows()
#cap.release()

##### Method 4 ####♦
#
#cap = cv2.VideoCapture(0)
#
###### Reshape Video frame ####
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
#
#while True:
#    
#    ret, frame = cap.read()
#    
#    # Define boundary rectangle containing the foreground object
#    height, width, _ = frame.shape
#    boundary_rectangle = (
#            int(width),
#            int(height),
#            int(width),
#            int(height))
#    
#    # gray scale image
#    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#    
#    nb_of_it = 10
#    # Binarized input image
#    
#    binarized_image = cv2.adaptiveThreshold(
#            gray,
#            maxValue=1,
#            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#            thresholdType=cv2.THRESH_BINARY,
#            blockSize=9,
#            C=7)
#    
#    # Initialize the mask with know information
#    mask = np.zeros((height, width), np.uint8)
#    mask[:] = cv2.GC_PR_BGD
#    mask[binarized_image == 0] = cv2.GC_FGD
#    
#    # Arrays used by the algorithm internally
#    background_model = np.zeros((1,65), np.float64)
#    foreground_model = np.zeros((1,65), np.float64)
#    
#    cv2.grabCut(
#            frame,
#            mask,
#            boundary_rectangle,
#            background_model,
#            foreground_model,
#            nb_of_it,
#            cv2.GC_INIT_WITH_MASK)
#    
#    grabcut_mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD),0 ,1).astype("uint8")
#    
#    segmented_image = frame.copy() * grabcut_mask[:, :, np.newaxis]
#
#
#    cv2.imshow('segmented_image', segmented_image)