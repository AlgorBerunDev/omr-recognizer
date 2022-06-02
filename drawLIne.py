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
def drawLine(image,p1,p2):
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
  cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), (0, 255, 0), 1)
  return image
def getIntersectPointPerpendiculars(startPoint, endPoint, point):
  x1 = startPoint[0]
  y1 = startPoint[1]
  x2 = endPoint[0]
  y2 = endPoint[1]
  xp = point[0]
  yp = point[1]
  sloppe = (y1 - y2) / (x1 - x2)
  m = -1 / sloppe
  x = (m * xp - yp - sloppe * x1 + y1) / (m - sloppe)
  y = m * x - m * xp + yp
  return (x, y)
print(getIntersectPointPerpendiculars(left_top_point, left_bottom_point, right_top_point))
# cv2.circle(im, getIntersectPointPerpendiculars(left_top_point, left_bottom_point, right_top_point), 2, (0, 255, 0), 2)
im = drawLine(im, right_top_point, right_top_point1)
im = drawLine(im, right_bottom_point, right_top_point)
im = drawLine(im, left_bottom_point, left_top_point)
cv2_imshow(im)