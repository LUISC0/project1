import cv2 as cv
import numpy as np
from PIL import Image


img = cv.imread('photos/CVtask.jpg')

#since the img is too big we scale it size down
scale_factor = 50

width = int(img.shape[1]*scale_factor/100)
height = int(img.shape[0]*scale_factor/100)
scaled = (width,height)

resized = cv.resize(img, scaled)

#converting the img to greyscale
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)


threshold, thresh = cv.threshold(gray , 225, 255, cv.THRESH_BINARY)

#finding the contours and drawing them for squares only
contours= cv.findContours(thresh, cv. RETR_LIST, cv.CHAIN_APPROX_NONE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cnt in contours:
    approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True)
    print(len(approx))
    if len(approx)==4:
        (x,y,w,h) = cv.boundingRect(approx)
        ar = w/float(h)
        if(ar>0.95 and ar<1.05):
            cv.drawContours(resized,[cnt],0,255,3)
cv.imshow('final',resized)

#cropping and resizing the markers
def rotate(img, angle, rotpoint=None):
    (height,width) = img.shape[:2]
    if rotpoint is None:
        rotpoint = (width//2,height//2)
    
    rotmat = cv.getRotationMatrix2D(rotpoint, angle, 1.0)
    dimensions = (width,height)
    
    return cv.warpAffine(img, rotmat, dimensions)

img1 = cv.imread('photos/HaHa.jpg')
haha =rotate(img1 , -15, None)    

img2 = cv.imread('photos/LMAO.jpg')
lmao =rotate(img2 , -15, None)

img3 = cv.imread('photos/XD.jpg')
xd =rotate(img3 , 13, None)

img4 = xd[80:533, 80:533]
img5 = lmao[80:533 , 80:533]
img6 = haha[80:533, 80:533]

cv.waitKey(0)
