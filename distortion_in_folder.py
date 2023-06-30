import cv2
import os
import numpy as np
from PIL import ImageTk, Image, ImageEnhance
vd_image = cv2.imread("C:/Users/Chuee/Desktop/diplom/test/Vid_distored.jpg", cv2.IMREAD_GRAYSCALE)
ir_image = cv2.imread("C:/Users/Chuee/Desktop/diplom/test/perspect.jpg", cv2.IMREAD_GRAYSCALE)
# ir_image = cv2.imread(r'./ir.jpg', cv2.IMREAD_GRAYSCALE)
vd_image = cv2.resize(vd_image, (1620, 720))
image_1 = vd_image[110:, 210:1100]
image_2 = ir_image[18:, 0:1100]
""" Метод максимума """

h, w = image_1.shape
for row in range(2, h - 2):
    for column in range(2, w - 2):
        if image_1[row, column] < image_2[row, column]:
            image_1[row, column] = image_2[row, column]

cv2.imshow('output_image',image_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
name = 'max_method_max'
full_name_gamma = new_path + name + '.jpg'
cv2.imwrite(full_name_gamma, image_1)



