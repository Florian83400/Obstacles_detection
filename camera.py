# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:51:04 2021

@author: ferna
"""
########### Obstacle Detection ###########

########### Importing Useful Libraries ###########
import numpy as np
import cv2


########### Functions of programm ###########

#### SIFT (Scale-Invariant Feature Transform) algorithm ####

def SIFT (image,nb_kp):
    sift = cv2.xfeatures2d.SIFT_create(nb_kp)  # Construct a SIFT object
    keypoint, des = sift.detectAndCompute(image, None)  # Find keypoints and descriptor
    image_final = cv2.drawKeypoints(image,keypoint,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)   # Drawkeypoints in circles
    return image_final, des, keypoint

#### Funtcion Brute-Force matcher ####
def BF(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2)   # To create the BFMatcher object with NORM_L2 is the euclidean distance
    matches = bf.knnMatch(des1,des2, k=2)   # Returns k best matches
    return matches


########### MAIN ###########    
upper_left = (50, 90)
bottom_right = (590, 390)

cap = cv2.VideoCapture(0)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #ret1, frame1 = cap.read()
    #ret2, frame2 = cap.read()
    
    
    # Stop to broadcast
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
    #Region of interest
    ROI = cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 2)
    rect_img = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]

    #Apply SIFT algorithm
    nb_kp = 100     # Number of keypoints that SIFT algorithm can detect
    img_SIFT,des, keypoint = SIFT(rect_img,nb_kp)
    img_SIFT2, des2 , keypoint2 = SIFT(rect_img, nb_kp)
    
    # Apply Brute-Force matcher
    bf = BF(des, des2)
    
    good_matches = []
    
    for m1, m2 in bf:
        good_matches.append(m1)  
    
    filtered_keypoints = []
    for match in good_matches:
        kp1 = keypoint[match.queryIdx].pt
        kp2 = keypoint2[match.queryIdx].pt
        
        filtered_keypoints.append(kp1)
    #print(len(filtered_keypoints))     
   
    kp_keeped = []    # List where keypoints filtered are saved
    for i in range(len(filtered_keypoints)):
        for keyPoint in keypoint:   #   Recover data from keypoints : coordinates and size
            if filtered_keypoints[i][0] == keyPoint.pt[0] and filtered_keypoints[i][1] == keyPoint.pt[1]:
                kp_keeped.append(keyPoint)
    
    kp_keeped = list(set(kp_keeped))    # Clears duplicates in the list
    #print(len(kp_keeped))
    
    size_kp = []
    for size in kp_keeped:
        s = size.size
        size_kp.append(s)
    #print(size_kp)
    
    mkp = []    # List of matched-filtered keypoints 
    for i in range(len(size_kp)-1):
        if size_kp[i+1]>size_kp[i]:
            mkp.append(size_kp[i+1])
    print(len(mkp))   
    
    final_list_kp = []
    # Recover keypoints about the size
    for i in range(len(mkp)):
        for size in kp_keeped:
            if mkp[i] == size.size:
                final_list_kp.append(size)
    
    #print(final_list_kp)
    
    # Print
    img_final = cv2.drawKeypoints(img_SIFT,final_list_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('final_frame with ROI and SIFT and matched keypoints',img_final)
#    cv2.imshow('Convex Hull results',drawing)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

    
    #Print
#    matched_img = cv2.drawMatches(img_SIFT,keypoint,img_SIFT, keypoint2, good_matches, None) # Draw matched keypoints between two frame
#    cv2.imshow('final_frame with ROI and SIFT and matched keypoints',matched_img) 
#   

    
    
#    coords_kp = []
#    for i in final_list_kp:
#        x = i.pt[0]
#        y = i.pt[1]
#        coords_kp.append([x,y])
#
#        
#    # Convex Hull to find Object of Interest
#    hull =[]
#    
#    # calculate points for each contour
#    for i in range(len(coords_kp)):
#        # creating convex hull object for each contour
#        hull.append(cv2.convexHull(np.array(coords_kp,dtype='float32'),False))
#        
#    drawing = np.zeros((img_SIFT.shape[0], img_SIFT.shape[1],3), dtype=np.uint8)
#    for i in range(len(hull)):
#        color = (255, 0, 0)
#        cv2.drawContours(drawing, hull , i,color, 1, 8)

    
