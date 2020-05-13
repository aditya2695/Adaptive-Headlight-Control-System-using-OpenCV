import timeit
import numpy as np 
import cv2 
import threading
import time

def getTemplate():
    template = cv2.imread('images/temp3.png',0)
    return template

def getTemplateDims():
    template=getTemplate()
    w,h = template.shape[::-1]
    return w,h

def getSourceVideo():
    #cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture('videos/mvk.avi')
    return cap

def getThreshold():
    threshold = 0.8715
    return threshold

def resizeFrame(img):
    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height) 
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized



def detectHeadlights(res,frame,img_bw1,pt):
    
    threshold = getThreshold()
    print(threshold,res)
    
    w,h = getTemplateDims() 

    img = resizeFrame(img_bw1)
     
    #avg color intensity of each row
    avg_color_per_row = np.average(img, axis=0)
    #avg color intensity of each frame
    avg_color = np.average(avg_color_per_row, axis=0)



    det=0
    
    if(avg_color< 0.19):
        print("high beam",avg_color)

        if np.any(res >=threshold):
            
            det=1;
        
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,205,255), 1)
            cv2.rectangle(img_bw1, pt, (pt[0] + w, pt[1] + h), (0,205,255), 1)
            return det
    
        elif np.any(res < threshold):
            det=0  
            return det
        

    else:
        print("low beam",avg_color)
        return 0 
    #print(np.array(res))

                    
    


cap  = getSourceVideo()
template=getTemplate()
while(getSourceVideo().isOpened()):
       
    ret, frame = cap.read()

    img =cv2.GaussianBlur(frame,(15,15), 0) 
    
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_bw2 = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 230).astype('uint8')
    
    ret3,img_bw1 = cv2.threshold(img_bw2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    res = cv2.matchTemplate(img_bw1,template,cv2.TM_CCOEFF_NORMED)

    threshold = getThreshold()
           
    loc = np.where(res >= .92*threshold)
             
    for pt in zip(*loc[::-1]):
                  
        if detectHeadlights(res,frame,img_bw1,pt)==1:
                    
           print("dim");

        elif detectHeadlights(res,frame,img_bw1,pt)==0:
                    
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

