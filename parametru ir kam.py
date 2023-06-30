import cv2
import numpy as np
import glob
# Load calibration images
# calibration_images = glob.glob("C:/Users/Chuee/Desktop/diplom/kalibrovka/ik/*.bmp")  # List of IR calibration images
# calibration_images = glob.glob("C:/Users/Chuee/Desktop/diplom/kalibrovka/vid/*.bmp")  # List of IR calibration images
calibration_images = glob.glob("C:/Users/Chuee/Desktop/diplom/3/4/*.bmp")  # List of IR calibration images
# calibration_images = glob.glob("C:/Users/Chuee/Desktop/diplom/t/*.bmp")  # List of IR calibration images


# Continue with further processing using the grayscale image 'gray'
# Prepare object points, assuming a 9x6 calibration pattern
pattern_size = (4, 7)
# pattern_size = (10, 6)
obj_points = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
obj_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
object_points = []  # 3D points in the real world
image_points = []   # 2D points in the image plane

# Detect calibration pattern corners in each image
for image in calibration_images:
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        object_points.append(obj_points)
        image_points.append(corners)

# Perform camera calibration
ret, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)

# # Compute rotation matrix from rotation vector
# rotation_matrix, _ = cv2.Rodrigues(rotation_vectors[0])
#
# # Compute projection matrix
# projection_matrix = intrinsic_matrix @ np.hstack((rotation_matrix, translation_vectors[0]))

# Print the obtained parameters
print("Intrinsic Matrix:")
print(intrinsic_matrix)

print("Distortion Coefficients:")
print(distortion_coeffs)

# print("Rotation Matrix:")
# print(rotation_matrix)
#
# print("Projection Matrix:")
# print(projection_matrix)