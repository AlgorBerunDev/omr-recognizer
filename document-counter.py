import cv2
from matplotlib import pyplot as plt

#initializing
img = cv2.imread("1.jpg")
img = cv2.resize(img, (int(480*2), int(640*2)))
# write code here
GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
BlurredFrame = cv2.GaussianBlur(GrayImg, (5, 5), 1)
CannyFrame = cv2.Canny(BlurredFrame, 190, 190)
contours, _ = cv2.findContours(CannyFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ContourFrame = img.copy()
ContourFrame = cv2.drawContours(ContourFrame, contours, -1, (255, 0, 255), 4)
CornerFrame = img.copy()
maxArea = 0
biggest = []
for i in contours :
    area = cv2.contourArea(i)
    if area > 1000 :
        peri = cv2.arcLength(i, True)
        edges = cv2.approxPolyDP(i, 0.02*peri, True)
        if area > maxArea and len(edges) == 4 :
            biggest = edges
            maxArea = area
if len(biggest) != 0 :
    CornerFrame = cv2.drawContours(CornerFrame, biggest, -1, (255, 0, 255), 25)
# resizing
img = cv2.resize(img, (480, 640))
GrayImg = cv2.resize(GrayImg, (480, 640))
BlurredFrame = cv2.resize(BlurredFrame, (480, 640))
CannyFrame = cv2.resize(CannyFrame, (480, 640))
ContourFrame = cv2.resize(ContourFrame, (480, 640))
CornerFrame = cv2.resize(CornerFrame, (480, 640))
#displaying
plt.imshow(img)
plt.imshow(GrayImg)
plt.imshow(BlurredFrame)
plt.imshow(CannyFrame)
plt.imshow(ContourFrame)
# plt.imshow(CornerFrame)
cv2.waitKey(0)