import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from google.colab.patches import cv2_imshow
import imutils


img = cv2.imread("4.jpg")
# Resize img width=Number, height=auto
img = imutils.resize(img, width=720)
original = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

ROI_number = 0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

maxArea = 0
biggest = []
for i in cnts :
    area = cv2.contourArea(i)
    if area > 1000 :
        peri = cv2.arcLength(i, True)
        edges = cv2.approxPolyDP(i, 0.1*peri, True)
        if area > maxArea and len(edges) == 4 :
            biggest = edges
            maxArea = area
if len(biggest) != 0 :
    CornerFrame = cv2.drawContours(img, biggest, -1, (0, 255, 0), 25)
    cv2_imshow(CornerFrame)
for cnt in cnts:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    print(len(approx))
    if len(approx)==5:
        print("Blue = pentagon")
        cv2.drawContours(img,[cnt],0,255,-1)
    # elif len(approx)==3:
    #     print("Green = triangle")
    #     cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    elif len(approx)==4:
        print("Red = square")
        cv2.drawContours(img,[cnt],0,(0,0,255),-1)
    # elif len(approx) == 6:
    #     print("Cyan = Hexa")
    #     cv2.drawContours(img,[cnt],0,(255,255,0),-1)
    # elif len(approx) == 8:
    #     print("White = Octa")
    #     cv2.drawContours(img,[cnt],0,(255,255,255),-1)
    # elif len(approx) > 12:
    #     print("Yellow = circle")
    #     cv2.drawContours(img,[cnt],0,(0,255,255),-1)
cv2_imshow(img)
cv2_imshow(thresh)