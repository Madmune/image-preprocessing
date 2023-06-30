from __future__ import print_function
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
#from common import splitfn
import os

img_names_undistort = [img for img in glob.glob('C:/Users/Chuee/Desktop/diplom/cam_Camera2_cam_20221111184048_2698926.bmp')]
new_path = 'C:/Users/Chuee/Desktop/diplom/test/'

camera_matrix = np.array([[378.91629, 0, 159.5],
                         [0, 378.91629, 119.5],
                         [0, 0, 1]]);
dist_coefs = np.array([-0.659404,  0.26198, 0,  0, 0]);

i = 0

#for img_found in img_names_undistort:
while i < len(img_names_undistort):
        img = cv2.imread(img_names_undistort[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 0, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        # crop and save the image
        x, y, w, h = roi
        # dst = dst[y:y+h-50, x+70:x+w-20]
        IMG = cv2.imshow("hhhh",dst)
        cv2.imshow('',img)
        name = img_names_undistort[i].split("/")
        name = name[5].split(".")
        name = name[0]
        full_name = new_path + name + '.jpg'

        #outfile = img_names_undistort + '_undistorte.bmp'
        print('Undistorted image written to: %s' % full_name)
        cv2.imwrite(full_name, dst)
        i = i + 1
cv2.waitKey(0)
cv2.destroyAllWindows()