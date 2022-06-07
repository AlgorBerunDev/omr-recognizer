from shapely.geometry import LineString
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from google.colab.patches import cv2_imshow
import imutils
import math


img = cv2.imread("6.jpg")
# Resize img width=Number, height=auto
img = imutils.resize(img, width=720)
original = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
BlurredFrame = cv2.GaussianBlur(gray, (11, 11), 1)
CannyFrame = cv2.Canny(BlurredFrame, 150, 150)
thresh = cv2.threshold(CannyFrame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

ROI_number = 0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

def getBiggestArea(cnts):
  maxArea = 0
  biggest = []
  for i in cnts :
      area = cv2.contourArea(i)
      if area > 1000 :
          peri = cv2.arcLength(i, True)
          edges = cv2.approxPolyDP(i, 0.05*peri, True)
          if area > maxArea and len(edges) == 4 :
              biggest = edges
              maxArea = area
  return biggest
biggest = getBiggestArea(cnts)

print("biggest", biggest)

if len(biggest) != 0 :
    CornerFrame = cv2.drawContours(img, biggest, -1, (0, 255, 0), 25)
    #cv2_imshow(CornerFrame)

def fillWithColor(cnts, img):
  for cnt in cnts:
      approx = cv2.approxPolyDP(cnt,0.05*cv2.arcLength(cnt,True),True)
      # print(len(approx))
      if len(approx)==5:
          # print("Blue = pentagon")
          cv2.drawContours(img,[cnt],0,255,-1)
      elif len(approx)==3:
          # print("Green = triangle")
          cv2.drawContours(img,[cnt],0,(0,255,0),-1)
      elif len(approx)==4:
          # print("Red = square")
          cv2.drawContours(img,[cnt],0,(0,0,255),-1)
      elif len(approx) == 6:
          # print("Cyan = Hexa")
          cv2.drawContours(img,[cnt],0,(255,255,0),-1)
      elif len(approx) == 8:
          # print("White = Octa")
          cv2.drawContours(img,[cnt],0,(255,128,255),-1)
      elif len(approx) > 12:
          # print("Yellow = circle")
          cv2.drawContours(img,[cnt],0,(0,255,255),-1)
fillWithColor(cnts, img)
# cv2_imshow(img)
# cv2_imshow(thresh)

def wrapDocument(biggest, original):
  points = biggest.reshape(4, 2)
  input_points = np.zeros((4, 2), dtype="float32")

  points_sum = points.sum(axis = 1)
  input_points[0] = points[np.argmin(points_sum)]
  input_points[3] = points[np.argmax(points_sum)]

  points_diff = np.diff(points, axis=1)
  input_points[1] = points[np.argmin(points_diff)]
  input_points[2] = points[np.argmax(points_diff)]

  (top_left, top_right, bottom_right, bottom_left) = input_points
  bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
  top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
  right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
  left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

  #Output image size
  max_width = max(int(bottom_width), int(top_width))
  # max_height = max(int(right_height), int(left_height))
  max_height = int(max_width * 1.414) # for A4

  #Desired points values in the output image
  converted_points = np.float32([[0,0], [max_width, 0], [0, max_height], [max_width, max_height]])

  # Perspective transformation
  matrix = cv2.getPerspectiveTransform(input_points, converted_points)
  return cv2.warpPerspective(original, matrix, (max_width, max_height))
print("biggest", type(biggest[0][0]))
img_wrapped = wrapDocument(biggest, original)
# cv2_imshow(img_wrapped)


gray = cv2.cvtColor(img_wrapped, cv2.COLOR_BGR2GRAY)
# BlurredFrame = cv2.GaussianBlur(gray, (15, 15), 1)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

ROI_number = 0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

fillWithColor(cnts, img_wrapped)
# cv2_imshow(gray)
# cv2_imshow(thresh)
# cv2_imshow(img_wrapped)











def getBiggestArea(cnts, canCheckLimitArea = False):
  maxArea = 0
  biggest = []
  for i in cnts :
      area = cv2.contourArea(i)
      if area > 100 or canCheckLimitArea:
          peri = cv2.arcLength(i, True)
          edges = cv2.approxPolyDP(i, 0.05*peri, True)
          print(len(edges))
          if area > maxArea and len(edges) == 4 :
              print(area)
              biggest = edges
              maxArea = area
  return biggest

image = gray.copy()
image2 = image.copy()
# [(left,top), (right,bottom)]
cordinatesOfReact = [
  [(0, 210), (100, 300)], # left-1
  [(620,210), (715,300)], # right-2
  [(620,10), (715,100)], # right-1
  [(620,750), (715,840)], # right-3
  [(0,750), (100,840)], # left-2
  [(620,850), (715,1011)] # right-4
]

for cordination in cordinatesOfReact:
  cv2.rectangle(image, cordination[0], cordination[1], (255, 0, 0), 0)
print(image.shape)
print(image2.shape)
# cv2_imshow(image)
sectionsMark = []
for cordination in cordinatesOfReact:
  top = cordination[0][1]
  bottom = cordination[1][1]
  left = cordination[0][0]
  right = cordination[1][0]
  sectionsMark.append(image2[top:bottom, left:right].copy())

sectionThreshs = []
for sectionMark in sectionsMark:
  sectionThreshs.append(cv2.threshold(sectionMark, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])

biggestCounters = []
for sectionThresh in sectionThreshs:
  cnts = cv2.findContours(sectionThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  biggest = getBiggestArea(cnts)
  if len(biggest) > 0:
    biggestArea = cv2.contourArea(biggest)
    biggestCounters.append([biggest, biggestArea, sectionThresh])

filledSections = []
for counter in biggestCounters:
  resultMark = cv2.cvtColor(counter[2], cv2.COLOR_GRAY2BGR)
  fillWithColor([counter[0]], resultMark)
#   cv2_imshow(resultMark)

# cv2_imshow(sectionThresh)
# cv2_imshow(sectionMark)



im = biggestCounters[0][2].copy()
print(biggestCounters[0][0])
# c = max(biggestCounters[0][0], key=cv2.contourArea)
c = biggestCounters[0][0]
# print(c)
def getExtremePoints(c, extreme):
  if extreme == 'left':
    return tuple(c[c[:, :, 0].argmin()][0])
  if extreme == 'right':
    return tuple(c[c[:, :, 0].argmax()][0])
  if extreme == 'top':
    return tuple(c[c[:, :, 1].argmin()][0])
  if extreme == 'bottom':
    return tuple(c[c[:, :, 1].argmax()][0])

y=getExtremePoints(biggestCounters[0][0], 'bottom')[1]
x=getExtremePoints(biggestCounters[0][0], 'right')[0]
y=y+cordinatesOfReact[0][0][1]
x=x+cordinatesOfReact[0][0][0]
left_top_point = (x, y) # left-1

y=getExtremePoints(biggestCounters[1][0], 'bottom')[1]
x=getExtremePoints(biggestCounters[1][0], 'left')[0]
y=y+cordinatesOfReact[1][0][1]
x=x+cordinatesOfReact[1][0][0]
right_top_point = (x, y) # right-2

y=getExtremePoints(biggestCounters[1][0], 'bottom')[1]
x=getExtremePoints(biggestCounters[1][0], 'right')[0]
y=y+cordinatesOfReact[1][0][1]
x=x+cordinatesOfReact[1][0][0]
right_top_point1 = (x, y) # right-2

y=getExtremePoints(biggestCounters[3][0], 'top')[1]
x=getExtremePoints(biggestCounters[3][0], 'left')[0]
y=y+cordinatesOfReact[3][0][1]
x=x+cordinatesOfReact[3][0][0]
right_bottom_point = (x, y) # right-3

y=getExtremePoints(biggestCounters[4][0], 'top')[1]
x=getExtremePoints(biggestCounters[4][0], 'right')[0]
y=y+cordinatesOfReact[4][0][1]
x=x+cordinatesOfReact[4][0][0]
left_bottom_point = (x, y) # left-2

print('left_top_point: ', left_top_point)
print('right_top_point: ', right_top_point)
print('left_bottom_point: ', left_bottom_point)
print('right_bottom_point: ', right_bottom_point)
im = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
cv2.circle(im, left_top_point, 2, (0, 255, 0), 2)
cv2.circle(im, right_top_point, 2, (0, 255, 0), 2)
cv2.circle(im, left_bottom_point, 2, (0, 255, 0), 2)
cv2.circle(im, right_bottom_point, 2, (0, 255, 0), 2)

### function to find slope 
def slope(p1,p2):
  x1,y1=p1
  x2,y2=p2
  if x2!=x1:
    return((y2-y1)/(x2-x1))
  else:
    return 'NA'

### main function to draw lines between two points
def drawLine(image,p1,p2, color = (0, 255, 0)):
  x1,y1=p1
  x2,y2=p2
  ### finding slope
  m=slope(p1,p2)
  ### getting image shape
  h,w=image.shape[:2]

  if m!='NA':
    ### here we are essentially extending the line to x=0 and x=width
    ### and calculating the y associated with it
    ##starting point
    px=0
    py=-(x1-0)*m+y1
    ##ending point
    qx=w
    qy=-(x2-w)*m+y2
  else:
    ### if slope is zero, draw a line with x=x1 and y=0 and y=height
    px,py=x1,0
    qx,qy=x1,h
  cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), color, 1)
  return image

im = drawLine(im, right_top_point, (left_top_point[0], right_top_point[1]))
im = drawLine(im, right_bottom_point, right_top_point)
im = drawLine(im, left_bottom_point, left_top_point)
# cv2_imshow(im)


im = gray.copy()
BlurredFrame = cv2.GaussianBlur(im, (21, 21), 1)
CannyFrame = cv2.Canny(BlurredFrame, 150, 150)
# cv2_imshow(CannyFrame)
thresh = cv2.threshold(CannyFrame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cv2_imshow(thresh)
# print(cnts)
min_x=left_top_point[0]
max_x=right_top_point[0]
min_y=right_top_point[1]
max_y=right_bottom_point[1]
im = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR)
y_tires=[]
x_tires=[]
extreme_points = []
for cnt in cnts:
  left = getExtremePoints(cnt, 'left')
  right = getExtremePoints(cnt, 'right')
  top = getExtremePoints(cnt, 'top')
  bottom = getExtremePoints(cnt, 'bottom')
  if left[0]>min_x and right[0] < max_x and top[1]>min_y and bottom[1] < max_y:
    area = cv2.contourArea(cnt)
    if area > 50 and area < 110:
      extreme_points.append([left, right, top, bottom])
      cv2.circle(im, left, 0, (0, 255, 0), 2)
      cv2.circle(im, right, 0, (0, 255, 0), 2)
      cv2.circle(im, top, 0, (0, 255, 0), 2)
      cv2.circle(im, bottom, 0, (0, 255, 0), 2)
      # im = drawLine(im, top, (0, top[1]))
      # im = drawLine(im, bottom, (0, bottom[1]))
      # im = drawLine(im, left, (left[0], 0))
      # im = drawLine(im, right, (right[0], 0))
      
      y_tires.append([top[1], bottom[1], 0])
      x_tires.append([left[0], right[0], 0])
im2 = im.copy()
fillWithColor(cnts, im2)
# cv2_imshow(im)
# cv2_imshow(im2)


def getLineIntersection1D(start1, end1, start2, end2, flag):
  if not(start1 < end2 and start2 < end1):
    return []
  line1_length = end1-start1
  line2_length = end2-start2
  if flag == 'min':
    min_start = 0
    min_end = 0
    if (start1<start2):
      min_start = start2
    else:
      min_start = start1
    
    if (end1<end2):
      min_end = end1
    else:
      min_end = end2
    intersection_length = min_end - min_start
    if (intersection_length < line1_length*0.4) or (intersection_length < line2_length*0.4):
      return []
    return [min_start, min_end]
  if flag == 'max':
    max_start = 0
    max_end = 0
    if (start1<start2):
      max_start = start1
    else:
      max_start = start2
    
    if (end1<end2):
      max_end = end2
    else:
      max_end = end1
    intersection_length = max_end - max_start
    if (intersection_length < line1_length*0.4) or (intersection_length < line2_length*0.4):
      return []
    return [max_start, max_end]

def getLinesIntersections(tires, tires_length):
  y_intersections= []
  i = 0
  while i < len(tires):
    if tires[i][2] > 0:
      i=i+1  
      continue
    current_intersection = [tires[i][0], tires[i][1]]
    j=i+1
    while j < len(tires):
      if tires[j][2] > 0:
        j=j+1
        continue
      line2 = tires[j]
      min_intersection = getLineIntersection1D(current_intersection[0], current_intersection[1], line2[0], line2[1], tires_length)
      if len(min_intersection) == 0:
        j=j+1
        continue
      else:
        current_intersection = min_intersection
        tires[j][2] = 1
      j=j+1
    y_intersections.append(current_intersection)
    i=i+1
  return y_intersections

y_intersections = getLinesIntersections(y_tires, 'min')
x_intersections = getLinesIntersections(x_tires, 'min')

y_im = gray.copy()
y_im = cv2.cvtColor(y_im, cv2.COLOR_GRAY2BGR)
for y_point in y_intersections:
  y_im = drawLine(y_im, (100, y_point[0]), (200, y_point[0]))
  y_im = drawLine(y_im, (100, y_point[1]), (200, y_point[1]), (255, 0, 255))
for x_point in x_intersections:
  y_im = drawLine(y_im, (x_point[0], 100), (x_point[0], 200))
  y_im = drawLine(y_im, (x_point[1], 100), (x_point[1], 200), (25, 0, 255))
# cv2_imshow(y_im)



def getMaxArea(cnts):
  maxArea = 0
  for i in cnts :
      area = cv2.contourArea(i)
      if area > maxArea:
          maxArea = area
  return maxArea

def checkWithExtremePoints(extreme_points, x1, y1, x2, y2):
  for points in extreme_points:
    left = points[0]
    right = points[1]
    top = points[2]
    bottom = points[3]
    if left[0] <= x1 and x2 <= right[0] and top[1] <= y1 and y2 <= bottom[1]:
      return True
  return False

table_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
answer_filled_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
ex = gray.copy()

y_intersections = sorted(y_intersections,key=lambda x: x[1])
x_intersections = sorted(x_intersections,key=lambda x: x[1])
i=0
cell_index=0
answers = []
for y in y_intersections:
  j=0
  for x in x_intersections:
    answer_range = 0
    if i >= 10 and j <= 11:
      j=j+1
      continue
    top_left_point = (x[0],y[0])
    bottom_right_point = (x[1],y[1])
    table_img = cv2.rectangle(table_img,top_left_point,bottom_right_point,(255,255,255),1)

    cell_img = ex[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]].copy()
    cell_img_thresh = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(cell_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    max_area = getMaxArea(cnts)
    shape_area = (bottom_right_point[0] - top_left_point[0])*(bottom_right_point[1] - top_left_point[1])
    if (shape_area)/2  < max_area:
      answer_range=answer_range+1
      result = checkWithExtremePoints(extreme_points, top_left_point[0], top_left_point[1], bottom_right_point[0], bottom_right_point[1])
      if result:
        answer_range=answer_range+1
      table_img = cv2.rectangle(table_img,top_left_point,bottom_right_point,(255,0,255),1)
    answers.append(answer_range)
    j=j+1
  i=i+1
# cv2_imshow(table_img)



answer_blocks = []
answer_sorted = []
for i in range(0,len(answers),4):
  variants = [answers[i], answers[i+1],answers[i+2],answers[i+3]]
  max_value = max(variants)
  max_index = variants.index(max_value)
  variant_response = ['A', 'B', 'C', 'D']
  if max_index < 4:
    answer_blocks.append(variant_response[max_index])
  else: 
    answer_blocks.append('NO')
index_of_answer = 0
for i in range(0, 50, 5):
  answer_sorted.append([index_of_answer+1, answer_blocks[i]])
  answer_sorted.append([index_of_answer+11, answer_blocks[i+1]])
  answer_sorted.append([index_of_answer+21, answer_blocks[i+2]])
  answer_sorted.append([index_of_answer+31, answer_blocks[i+3]])
  answer_sorted.append([index_of_answer+61, answer_blocks[i+4]])
  print(
      str(index_of_answer+1) + answer_blocks[i]+" "+
      str(index_of_answer+11) + answer_blocks[i+1]+" "+
      str(index_of_answer+21) + answer_blocks[i+2]+" "+
      str(index_of_answer+31) + answer_blocks[i+3]+" "+
      str(index_of_answer+61) + answer_blocks[i+4]
  )
  index_of_answer=index_of_answer+1

index_of_answer = 41
for i in range(50, len(answer_blocks), 2):
  answer_sorted.append([index_of_answer, answer_blocks[i]])
  answer_sorted.append([index_of_answer+30, answer_blocks[i+1]])
  print(
      str(index_of_answer) + answer_blocks[i]+" "+
      str(index_of_answer+30) + answer_blocks[i+1]+" "
  )
  index_of_answer=index_of_answer+1
answer_sorted = sorted(answer_sorted, key=lambda x: x[0])
print(answer_sorted)