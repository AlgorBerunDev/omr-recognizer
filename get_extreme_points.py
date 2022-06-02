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

y=getExtremePoints(biggestCounters[3][0], 'top')[1]
x=getExtremePoints(biggestCounters[3][0], 'left')[0]
y=y+cordinatesOfReact[3][0][1]
x=x+cordinatesOfReact[3][0][0]
left_bottom_point = (x, y) # right-3

y=getExtremePoints(biggestCounters[4][0], 'top')[1]
x=getExtremePoints(biggestCounters[4][0], 'right')[0]
y=y+cordinatesOfReact[4][0][1]
x=x+cordinatesOfReact[4][0][0]
right_bottom_point = (x, y) # left-2

print('left_top_point: ', left_top_point)
print('right_top_point: ', right_top_point)
print('left_bottom_point: ', left_bottom_point)
print('right_bottom_point: ', right_bottom_point)
im = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
cv2.circle(im, left_top_point, 2, (0, 255, 0), 2)
cv2.circle(im, right_top_point, 2, (0, 255, 0), 2)
cv2.circle(im, left_bottom_point, 2, (0, 255, 0), 2)
cv2.circle(im, right_bottom_point, 2, (0, 255, 0), 2)
cv2_imshow(im)