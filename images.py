import cv2
import imutils
import numpy as np

puzzle = cv2.imread('images/puzzle.jpg')
img_gray = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)

template = cv2.imread('images/waldotest.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
(_, _, minLoc, maxLoc) = cv2.minMaxLoc(res)
threshold = 0.9
loc = np.where( res >= threshold)

# grab the bounding box of waldo and extract him from
# the puzzle image
topLeft = maxLoc
botRight = (topLeft[0] + w, topLeft[1] + h)
roi = puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]
 
# construct a darkened transparent 'layer' to darken everything
# in the puzzle except for waldo
mask = np.zeros(puzzle.shape, dtype = "uint8")
puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

# put the original waldo back in the image so that he is
# 'brighter' than the rest of the image
puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi
 
# display the images
cv2.imshow("Puzzle", imutils.resize(puzzle, height = 650))
cv2.imshow("Waldo", template)
cv2.waitKey(0)
