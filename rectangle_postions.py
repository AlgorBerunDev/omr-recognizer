def getBiggestArea(cnts):
  maxArea = 0
  biggest = []
  for i in cnts :
      area = cv2.contourArea(i)
      if area > 100 :
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
  [(0, 210), (100, 300)],
  [(620,210), (715,300)],
  [(620,10), (715,100)],
  [(620,750), (715,840)],
  [(0,750), (100,840)],
  [(620,850), (715,1011)]
]

for cordination in cordinatesOfReact:
  cv2.rectangle(image, cordination[0], cordination[1], (255, 0, 0), 2)
print(image.shape)
cv2_imshow(image)
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
  cv2_imshow(resultMark)

# cv2_imshow(sectionThresh)
# cv2_imshow(sectionMark)
