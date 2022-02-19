import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
print("Test")
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
print("Test")

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    cv.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
        template = frame
        break

  # Break the loop
  else:
    break

cv.imwrite('test.jpg', template)

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()

img = cv.imread('puzzle.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img2 = img.copy()
#template = cv.imread('template.jpg')

hsv = cv.cvtColor(template, cv.COLOR_BGR2RGB)

lower_red = np.array([75, 140, 120])
upper_red = np.array([135,200,185])

mask = cv.inRange(hsv, lower_red, upper_red)
res = cv.bitwise_and(template, template, mask=mask)

# plt.imshow(res)
# plt.show()

gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
hh, ww = thresh.shape

# make bottom 2 rows black where they are white the full width of the image
thresh[hh-3:hh, 0:ww] = 0

# get bounds of white pixels
white = np.where(thresh==255)
xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
print(xmin,xmax,ymin,ymax)

# crop the image at the bounds adding back the two blackened rows at the bottom
crop = template[ymin:ymax-3, xmin:xmax]

img_width = img.shape[1]
template_dim = (img_width * 2) / 30.0

crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
crop = cv.resize(crop, (int(template_dim), int(template_dim)))
# plt.imshow(crop)
# plt.show()

template = crop

lower_red = np.array([65, 130, 110])
upper_red = np.array([145,210,195])

mask = cv.inRange(template, lower_red, upper_red)
res = cv.bitwise_and(template, template, mask=mask)

# plt.imshow(res)
# plt.show()

gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
hh, ww = thresh.shape

# make bottom 2 rows black where they are white the full width of the image
thresh[hh:hh, 0:ww] = 0

thresh = 255-thresh
kernel = np.ones((6,6),np.uint8)
thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
# plt.imshow(thresh)
# plt.show()

thresh = thresh[30:thresh.shape[0]-30, 30:thresh.shape[0]-30]
template = template[30:template.shape[0]-30, 30:template.shape[0]-30]

plt.imshow(thresh)
plt.show()
plt.imshow(template)
plt.show()
# get bounds of white pixels
white = np.where(thresh==255)
xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
print(xmin,xmax,ymin,ymax)

# crop the image at the bounds adding back the two blackened rows at the bottom
crop = template[ymin+4:ymax-2, xmin+4:xmax-2]

#crop = crop[40:crop.shape[0]-40, 40:crop.shape[1]-40]

plt.imshow(crop)
plt.show()
template = crop

max_match = 0
locs = []
val = []
for i in range(4):
    w, h = template.shape[::-1][1:]
    res = cv.matchTemplate(img,template,cv.TM_CCOEFF_NORMED)
    loc = np.where(res > 0.43)
    if np.amax(res) > max_match:
        max_match = np.amax(res)
    result = np.where(res == np.amax(res))
    locs.append(list(zip(result[0], result[1])))
    val.append(np.amax(res))
    for pt in zip(*loc[::-1]):
        cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
    template = cv.rotate(template, cv.ROTATE_90_CLOCKWISE)

# print(locs)
# print(val)
#
# locs = locs[val.index(max(val))][0]
# print(locs)
# cv.rectangle(img, locs, (locs[0] + w, locs[1] + h), (0, 255, 0), 2)
plt.imshow(img)
plt.show()