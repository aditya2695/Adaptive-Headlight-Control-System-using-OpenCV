import timeit
import cv2
import numpy as np    
import threading
import time



template = cv2.imread('images/temp3.png',0)
cap = cv2.VideoCapture('videos/mvk.avi')
#cap = cv2.VideoCapture(1)
w,h = template.shape[::-1]
print(w)
print(h) 
 

while(1):
    
    
    ret, frame = cap.read()

    img =cv2.GaussianBlur(frame,(15,15), 0) 
    
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_bw2 = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 230).astype('uint8')
    
    ret3,img_bw1 = cv2.threshold(img_bw2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    res = cv2.matchTemplate(img_bw1,template,cv2.TM_CCOEFF_NORMED)

    threshold = 0.8715
    
    #avg color intensity of each row
    avg_color_per_row = np.average(img_bw1, axis=0)
    #avg color intensity of each frame
    avg_color = np.average(avg_color_per_row, axis=0)
    
    if(avg_color< 0.19):
        print("high beam",avg_color)

    else:
        print("low beam",avg_color)
            
    loc = np.where( res >= .92*threshold)
             
    for pt in zip(*loc[::-1]):
        def detect1():
                    
            if np.any(res >=threshold):
                
                det=1;
                
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,205,255), 1)
                cv2.rectangle(img_bw1, pt, (pt[0] + w, pt[1] + h), (0,205,255), 1)
                return det
            
            elif np.any(res < threshold):
                        
                
                det=0;  
                return det           

        if detect1()==1:
                    
           print("dim");

        elif detect1()==0:
                    
           print("bright");
    
    #To Display Thresholded video        
    cv2.imshow('Detected1',img_bw1)
    k = cv2.waitKey(10) & 0xff
        # ASCII 27 IS ESC key 
    if k==27 :
        break
    #To Display detected frame               
    cv2.imshow('Detected',frame)
    k = cv2.waitKey(10) & 0xff
        # ASCII 27 IS ESC key
    if k==27 :
        break

cap.release()
cv2.destroyAllWindows()

