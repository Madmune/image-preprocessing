import cv2
import numpy as np

# Load the IR image
img = cv2.imread('C:/Users/Chuee/Desktop/diplom/test/testtest.jpg', 0)

# Binarize the image using adaptive thresholding
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological opening to remove noise
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Apply morphological closing to fill holes inside objects
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Display the results
cv2.imshow('Original', img)
cv2.imshow('Thresholded', thresh)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()