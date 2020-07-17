import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image', required=True, help='path to the image file')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
# cv2.waitKey(1000)

# gradX = cv2.Sobel(gray, ddepth= cv2.cv.CV_32F, dx=1, dy=0, Ksize=-1)
# gradY = cv2.Sobel(gray, ddepth= cv2.cv.CV_32F, dx=0, dy=1, Ksize=-1)
gradX = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
gradY = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow('gradient', gradient)
# cv2.waitKey(1000)

blurred = cv2.blur(gradient, (9,9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
cv2.imshow('blurred', thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)



closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

cv2.imshow('closed', closed)

(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]


rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

cv2.drawContours(image, [box], -1, (0,255,0), 3)
cv2.imshow('Image', image)

cv2.waitKey(0)

