import cv2
import numpy as np
# #('C:/Users/Chuee/Desktop/diplom/Camera_Camera_01_Camera_20221024160312_4123650.png')
# #('C:/Users/Chuee/Desktop/diplom/Camera_Camera_02_Camera_20221024160315_4126116.bmp')
# # Load the larger image
# img1 = cv2.imread('C:/Users/Chuee/Desktop/diplom/test/testvidtest.jpg')
#
# # Load the smaller image
# img2 = cv2.imread('C:/Users/Chuee/Desktop/diplom/test/cam_Camera2_cam_20221111184048_2698926.jpg')
#
# # Load the IR image
# img_ir = cv2.imread('C:/Users/Chuee/Desktop/diplom/test/cam_Camera2_cam_20221111184048_2698926.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Apply median filter to enhance the image
# img_ir_filtered = cv2.medianBlur(img_ir, 5)
#
# # Find key points in the IR image
# detector = cv2.ORB_create()
# keypoints_ir, descriptors_ir = detector.detectAndCompute(img_ir_filtered, None)
#
# # Load the visible image
# img_visible = cv2.imread('C:/Users/Chuee/Desktop/diplom/test/testvidtest.jpg', cv2.IMREAD_COLOR)
#
# # Find key points in the visible image
# keypoints_visible, descriptors_visible = detector.detectAndCompute(img_visible, None)
#
# # Match the key points between the IR and visible images
# matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = matcher.match(descriptors_ir, descriptors_visible)
#
# # Apply affine transformation using the matched key points
# src_pts = np.float32([keypoints_ir[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([keypoints_visible[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#
# M, mask = cv2.estimateAffine2D(src_pts, dst_pts)
#
# # Apply the affine transformation to the IR image
# img_ir_aligned = cv2.warpAffine(img_ir_filtered, M, (img_visible.shape[1], img_visible.shape[0]))
#
# # Display the results
# cv2.imshow('IR Image', img_ir_filtered)
# cv2.imshow('Aligned IR Image', img_ir_aligned)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
