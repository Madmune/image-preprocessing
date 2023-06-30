import tkinter as tk
from tkinter import filedialog,ttk
from tkinter import *
from PIL import ImageTk, Image, ImageEnhance
import cv2
import numpy as np
from tkinter.messagebox import showerror, showwarning, showinfo
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

root = tk.Tk()
root.geometry("1280x920")

img1, img2 = None, None
# Define function to upload image
def upload_image():
    # Get the filename from the user
    global filename
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All Files", "*.*")))
    # print(filename)
    # Read the image using OpenCV
    img1 = Image.open(filename)
    img1 = cv2.imread(f'{filename}')
    # cv2.imshow("original image",img1)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img1 = cv2.resize(img1, (320, 320))
    img1 = Image.fromarray(img1)
    photo1 = ImageTk.PhotoImage(image=img1)
    # lb = tk.Label(root, text=' original img')
    # lb.place(x=500, y=10)
    lt_img.configure(image=photo1)
    lt_img.image = photo1

def upload_image1():
    # Get the filename from the user
    global filename
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All Files", "*.*")))
    # Read the image using OpenCV
    img = cv2.imread(f'{filename}')
    # cv2.imshow("original image", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    # lb = tk.Label(root, text=' original img')
    # lb.place(x=500, y=310)
    rt_img.configure(image=photo1)
    rt_img.image = photo1

def distors_ik():
    global dst
    img_names_undistort = f'{filename}'
    # print(img_names_undistort)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'

    # camera_matrix = np.array([[378.91629, 0, 159.5],
    #                           [0, 378.91629, 119.5],
    #                           [0, 0, 1]]);
    #
    # dist_coefs = np.array([-0.659404, 0.26198, 0, 0, 0]);

    camera_matrix = np.array([[1.34865872e+03, 0.00000000e+00, 6.57125529e+02],
 [0.00000000e+00, 1.00469281e+03, 3.23746371e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_coefs = np.array([-4.68112416e-01, 2.02520045e-01, -3.74472342e-03,  -2.22472197e-02, 8.5388035e-01])
  #   dist_coefs = np.array([-3.21109854e+00 , 9.97555313e+01, -3.31430729e-02,  1.24350401e-03,
  # -1.21368586e+03])



    i = 0

    # for img_found in img_names_undistort:
    while i < len(img_names_undistort):
        img = cv2.imread(img_names_undistort)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 0, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        # crop and save the image
        x, y, w, h = roi
        # dst = dst[y:y+h-50, x+70:x+w-20]

        # IMG = cv2.imshow("img with out distorsion", dst)

        name = 'IR_distored'
        name1 = 'resize'
        full_name_IR = new_path + name + '.jpg'
        full_name_IR1 = new_path + name1 + '.jpg'
        i = i + 1
    resize_img =cv2.resize(dst, (1193,902), interpolation = cv2.INTER_AREA)
    cv2.imwrite(full_name_IR, dst)
    cv2.imwrite(full_name_IR1, resize_img)
    # cv2.imshow("img with out", resize_img)

    img = cv2.resize(dst, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    lb_img.configure(image=photo1)
    lb_img.image = photo1

def distors_vid():

    img_vid_undistort = f'{filename}'
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'

    # camera_matrix = np.array([[1.35455917e+03, 0, 9.99707081e+02],
    #                           [0, 1.35511329e+03, 5.76876742e+02],
    #                           [0, 0, 1]]);

    camera_matrix = np.array([[872.36119432, 0, 676.89776845],
                              [0, 873.84266618, 393.41918832],
                              [0, 0, 1]]);
    dist_coefs = np.array([-0.44876128, 0.37527417, -0.0009433, -0.00097537, -0.48784801]);

    i = 0

    # for img_found in img_names_undistort:
    while i < len(img_vid_undistort):
        img = cv2.imread(img_vid_undistort)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 0, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        # crop and save the image
        x, y, w, h = roi
        # dst = dst[y:y+h-50, x+70:x+w-20]
        # IMG = cv2.imshow("img with out distorsion", dst)
        name = 'Vid_distored'
        full_name_vid = new_path + name + '.jpg'
        outfile = img_vid_undistort + '_undistorte.jpg'

        i = i + 1
    cv2.imwrite(full_name_vid, dst)
    image = Image.open(full_name_vid)
    # Get the width and height of the image
    width, height = image.size
    #
    # # Calculate the crop box dimensions (left, upper, right, lower)
    # crop_box = ((width - 1193) // 2, (height - 902) // 2, (width + 1193) // 2,
    #             (height + 902) // 2)  # set the crop box size as required
    #
    # # # Crop the image using the crop box
    # cropped_image = image.crop(crop_box)
    # image = cropped_image.save(f"{new_path}/croped_vid.jpg")
    # cropped_image.show()
    # cropped_image= np.array(cropped_image)
    img = cv2.resize(dst, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    rb_img.configure(image=photo1)
    rb_img.image = photo1

def rotate():
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                          filetypes=(("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All Files", "*.*")))
    # Read the image using OpenCV
    img = Image.open(f'{filename}')
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    rot = int(rota.get())
    rotated_image = img.rotate(rot)
    name = 'rotated_image'
    full_name_high_pass = new_path + name + '.jpg'
    rotated_image=rotated_image.save(full_name_high_pass)

    img = np.array(img)
    img = cv2.resize(img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    lt_img.configure(image=photo1)
    lt_img.image = photo1

    rotated_image = np.array(rotated_image)
    rotated_image = cv2.resize(rotated_image, (320, 320))
    rotated_image = Image.fromarray(rotated_image)
    photo1 = ImageTk.PhotoImage(image=rotated_image)
    lb_img.configure(image=photo1)
    lb_img.image = photo1

def gauss_img():
    file = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img = cv2.imread(file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Apply a Gaussian filter with kernel size 5 and standard deviation 0/
    k_gauss = int(kern_gauss.get())
    img_blur = cv2.GaussianBlur(img, (k_gauss, k_gauss), 0)

    # Display the original and filtered images
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Blurred Image', img_blur)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'gauss_img'
    full_name_gauss = new_path + name + '.jpg'
    cv2.imwrite(full_name_gauss, img_blur)
    #
    # img = np.array(img)
    # img_blur = np.array(img_blur)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    rt_img.configure(image=photo1)
    rt_img.image = photo1

    img = cv2.resize(img_blur, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    rb_img.configure(image=photo1)
    rb_img.image = photo1

def laplc_img():
    file = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Apply Laplacian filter with kernel size 3
    k = int(kern.get())
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=k)
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Blurred Image', laplacian)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'laplacian_img'
    full_name_laplacian = new_path + name + '.jpg'
    cv2.imwrite(full_name_laplacian, laplacian)

    img = cv2.resize(img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    lt_img.configure(image=photo1)
    lt_img.image = photo1


    laplacian = cv2.resize(laplacian, (320, 320))
    laplacian = Image.fromarray((laplacian).astype(np.uint8))
    photo1 = ImageTk.PhotoImage(image=laplacian)
    lb_img.configure(image=photo1)
    lb_img.image = photo1

def sredniy():

    file = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img = cv2.imread(file)
    k = int(sred.get())
    kernel_sred = np.ones((k, k), np.float32) / 9
    processed_image = cv2.filter2D(img, -1, kernel_sred)

    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'sredniy'
    full_name_high_pass = new_path + name + '.jpg'
    cv2.imwrite(full_name_high_pass, processed_image)
    # img = np.array(img)
    # im_output = np.array(im_output)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    rt_img.configure(image=photo1)
    rt_img.image = photo1

    img = cv2.resize(processed_image, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    rb_img.configure(image=photo1)
    rb_img.image = photo1

def median():
    file = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img = cv2.imread(file)
    # Apply a median filter with kernel size 5
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    median = int(m.get())
    img_median = cv2.medianBlur(img, median)

    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'median'
    full_name_median = new_path + name + '.jpg'
    cv2.imwrite(full_name_median, img_median)

    img = cv2.resize(img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    lt_img.configure(image=photo1)
    lt_img.image = photo1

    img = cv2.resize(img_median, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    lb_img.configure(image=photo1)
    lb_img.image = photo1

def homography():
    # Load the input image
    showinfo(title="Предупреждение", message="Загрузите ИК изображение")
    file_ir = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img_ir = cv2.imread(file_ir)

    # Apply a filter to enhance the IR image
    img_ir_filtered = cv2.bilateralFilter(img_ir, 9, 75, 75)

    # Convert the filtered image to grayscale
    gray_ir = cv2.cvtColor(img_ir_filtered, cv2.COLOR_BGR2GRAY)

    # Load the feature detector and descriptor
    detector = cv2.ORB_create()

    # Find the keypoints and descriptors in the IR image
    keypoints_ir, descriptors_ir = detector.detectAndCompute(gray_ir, None)

    # Load the second input image
    showinfo(title="Предупреждение", message="Загрузите видимое изображение")

    file_vid = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                         filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img_visible = cv2.imread(file_vid)

    # Convert the second image to grayscale
    gray_visible = cv2.cvtColor(img_visible, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors in the second image
    keypoints_visible, descriptors_visible = detector.detectAndCompute(gray_visible, None)

    # Match the keypoints in both images
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_ir, descriptors_visible)

    # Sort the matches by their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Choose the best 10% of the matches
    num_good_matches = int(len(matches) * 0.1)
    good_matches = matches[:num_good_matches]

    # Extract the keypoint coordinates for the good matches in both images
    points_ir = np.zeros((len(good_matches), 2), dtype=np.float32)
    points_visible = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points_ir[i, :] = keypoints_ir[match.queryIdx].pt
        points_visible[i, :] = keypoints_visible[match.trainIdx].pt

    # Find the homography matrix that maps the IR image to the second image
    H, _ = cv2.findHomography(points_ir, points_visible, cv2.RANSAC)

    # Apply the homography to the IR image
    warped_ir = cv2.warpPerspective(img_ir, H, (img_visible.shape[1], img_visible.shape[0]))

    # Display the results
    cv2.imshow('Filtered IR Image', img_ir_filtered)
    cv2.imshow('Warped IR Image', warped_ir)

def points_sift():

    showinfo(title="Предупреждение", message="Выберите изображение для аффиного преобразования")
    file = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                         filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img = cv2.imread(file)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect keypoints using SIFT
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)

    # Draw keypoints on the image
    img_keypoints = cv2.drawKeypoints(img, keypoints, None)

    # Define the affine transformation matrix
    angle = 45
    scale = 1.5
    tx = 50
    ty = -20
    M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    # Apply affine transformation to the keypoints
    pts = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
    transformed_pts = cv2.transform(pts, M)

    # Draw the transformed keypoints on the image
    img_transformed = cv2.drawKeypoints(img, [cv2.KeyPoint(pt[0][0], pt[0][1], 1) for pt in transformed_pts], None,
                                        color=(0, 255, 0))


    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'affin_img'
    full_name_affin = new_path + name + '.jpg'
    cv2.imwrite(full_name_affin, img_transformed)
    # Display the images
    cv2.imshow('Original', img_keypoints)
    cv2.imshow('Transformed', img_transformed)

    def find_keypoints(image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_keypoints(descriptors1, descriptors2):
        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors of the two images
        matches = bf.match(descriptors1, descriptors2)

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        return matches

    # Load the infrared and visible images
    file = filedialog.askopenfilename(initialdir="/", title="Select Image",
                               filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    file1 = filedialog.askopenfilename(initialdir="/", title="Select Image",
                               filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))

    infrared_image = cv2.imread(file)
    # infrared_image = cv2.cvtColor(infrared_image, cv2.COLOR_BGR2GRAY)
    visible_image = cv2.imread(file1)
    # visible_image = cv2.cvtColor(visible_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('',visible_image)
    # Find keypoints and compute descriptors for both images
    infrared_keypoints, infrared_descriptors = find_keypoints(infrared_image)
    visible_keypoints, visible_descriptors = find_keypoints(visible_image)

    # Match keypoints between the two images
    matches = match_keypoints(infrared_descriptors, visible_descriptors)

    # Draw the matched keypoints on the images
    matched_image = cv2.drawMatches(infrared_image, infrared_keypoints, visible_image, visible_keypoints, matches[:10],
                                    None)

    # Display the result
    cv2.imshow('Matched Keypoints', matched_image)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'matched_image'
    full_name_gauss = new_path + name + '.jpg'
    cv2.imwrite(full_name_gauss, matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dilate():
    file = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img = cv2.imread(file)
    morph_kernel = np.ones((3, 3))
    dilate_img = cv2.erode(img, kernel= morph_kernel, iterations=1)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'dilate'
    full_name_dilate = new_path + name + '.jpg'
    cv2.imwrite(full_name_dilate, dilate_img)
    # Display the images
    cv2.imshow('Original', img)
    cv2.imshow('Transformed', dilate_img)

def IR_homograh():
    def mouse_handler(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(data['im'], (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", data['im'])
            if len(data['points']) & 4: data['points'].append([x, y])


    if __name__ == '__main__':
        # Read in the image.
        showinfo(title="Предупреждение", message="Выберите ИК изображение для гомографии")
        file = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                          filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
        im_src = cv2.imread(file)
        # im_src = cv2.imread("C:/Users/Chuee/Desktop/diplom/test/IR_distored.jpg")

        # Show image and wait for 4 clicks.
        pts_src = np.float32([[356, 236], [970, 253], [943, 686], [348, 623]])

        # Book size
        size = (687, 584)

        # Destination coordinates located in the center of the image
        pts_dst = np.float32([[im_src.shape[1] / 2 - size[0] / 2, im_src.shape[0] / 2 - size[1] / 2],
                              [im_src.shape[1] / 2 + size[0] / 2, im_src.shape[0] / 2 - size[1] / 2],
                              [im_src.shape[1] / 2 + size[0] / 2, im_src.shape[0] / 2 + size[1] / 2],
                              [im_src.shape[1] / 2 - size[0] / 2, im_src.shape[0] / 2 + size[1] / 2]])
        # print(pts_src)
        # Calculate the homography
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        # Warp source image to destination
        im_out = cv2.warpPerspective(im_src, M, (im_src.shape[1], im_src.shape[0]))
        # cv2.imshow('',im_out)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'warped'
    full_name_warped = new_path + name + '.jpg'
    cv2.imwrite(full_name_warped, im_out)

    img = cv2.resize(im_src, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    rt_img.configure(image=photo1)
    rt_img.image = photo1

    img = cv2.resize(im_out, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    rb_img.configure(image=photo1)
    rb_img.image = photo1

def contrast():
    file_ir = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                         filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img = Image.open(file_ir)
    ench = ImageEnhance.Contrast(img)
    fac = float(f.get())
    contrasred_img = ench.enhance(fac)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'contrasred_img'
    full_name = new_path + name + '.jpg'
    # contrasred_img.show()
    # img.show()
    contrasred_img.save(full_name)

    img = np.array(img)
    contrasred_img = np.array(contrasred_img)
    img = cv2.resize(img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    rt_img.configure(image=photo1)
    rt_img.image = photo1

    img = cv2.resize(contrasred_img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    rb_img.configure(image=photo1)
    rb_img.image = photo1

def rezkost():
    file_ir = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                         filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img = Image.open(file_ir)
    ench = ImageEnhance.Sharpness(img)
    fac = int(rez.get())
    shareped_img = ench.enhance(fac)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'sharped'
    full_name = new_path + name + '.jpg'
    shareped_img.save(full_name)

    img = np.array(img)
    img = cv2.resize(img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    rt_img.configure(image=photo1)
    rt_img.image = photo1

    shareped_img = np.array(shareped_img)
    shareped_img = cv2.resize(shareped_img, (320, 320))
    shareped_img = Image.fromarray(shareped_img)
    photo1 = ImageTk.PhotoImage(image=shareped_img)
    rb_img.configure(image=photo1)
    rb_img.image = photo1

def brightness():
    file_ir = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                         filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img = Image.open(file_ir)
    enhancer = ImageEnhance.Brightness(img)
    factor = float(br.get())
    im_output = enhancer.enhance(factor)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'brightness'
    full_name = new_path + name + '.jpg'
    im_output.save(full_name)

    img = np.array(img)
    im_output = np.array(im_output)
    img = cv2.resize(img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    lt_img.configure(image=photo1)
    lt_img.image = photo1

    img = cv2.resize(im_output, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    lb_img.configure(image=photo1)
    lb_img.image = photo1

def gamma_korection():
    file = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gamma =float(gam.get())

    # gamma = 2
    gamma_img = np.power(img/float(np.max(img)), gamma)
    gamma_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'gamma_korection1'
    full_name_gamma = new_path + name + '.jpg'
    cv2.imwrite(full_name_gamma, gamma_img)

    # gamma_img = np.array(gamma_img)
    img = cv2.resize(img, (320, 320))
    img = Image.fromarray(img)
    photo1 = ImageTk.PhotoImage(image=img)
    lt_img.configure(image=photo1)
    lt_img.image = photo1

    gamma_img = cv2.resize(gamma_img, (320, 320))
    # gamma_img = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB)
    gamma_img = Image.fromarray(gamma_img)
    # gamma_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    photo1 = ImageTk.PhotoImage(image=gamma_img)
    lb_img.configure(image=photo1)
    lb_img.image = photo1

def mast_method():
    vd_image = cv2.imread("C:/Users/Chuee/Desktop/diplom/test/Vid_distored.jpg",cv2.IMREAD_GRAYSCALE)

    ir_image = cv2.imread("C:/Users/Chuee/Desktop/diplom/test/contrasred_img.jpg",cv2.IMREAD_GRAYSCALE)
    # ir_image = cv2.imread(r'./ir.jpg', cv2.IMREAD_GRAYSCALE)
    image_1 = vd_image[137:932, 13:1171]
    image_2 = ir_image[0:785, 47:1205]
    output_image = image_1
    h, w = output_image.shape
    threshold = 145
    for row in range(2, h - 2):
        for column in range(2, w - 2):
            if int(image_2[row, column]) >= threshold:
                output_image[row, column] = image_2[row, column]

    cv2.imshow('output_image',output_image)
    new_path = 'C:/Users/Chuee/Desktop/diplom/test/'
    name = 'max_method'
    full_name_gamma = new_path + name + '.jpg'
    cv2.imwrite(full_name_gamma, output_image)

def rmse():
    def calculate_rmse(image1_path, image2_path):
        # Open the images
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # Convert the images to grayscale
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        # Convert the images to NumPy arrays
        array1 = np.array(image1)
        array2 = np.array(image2)

        # Calculate the squared error between the images
        squared_error1 = np.square(array1)
        squared_error2 = np.square(array2)

        # Calculate the mean squared error
        mse1 = np.mean(squared_error1)
        mse2 = np.mean(squared_error2)

        # Calculate the root mean squared error
        rmse1 = np.sqrt(mse1)
        rmse2 = np.sqrt(mse2)

        return rmse1, rmse2

    def plot_standard_deviation(image1_path, image2_path):
        # Open the images
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # Convert the images to grayscale
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        # Convert the images to NumPy arrays
        array1 = np.array(image1)
        array2 = np.array(image2)
        # Calculate the standard deviation of the images
        std_deviation1 = np.std(array1)
        std_deviation2 = np.std(array2)

        # Create a bar plot of the standard deviation values
        labels = ['Image 1', 'Image 2']
        std_deviations = [std_deviation1, std_deviation2]

        plt.bar(labels, std_deviations)
        plt.ylabel('Standard Deviation')
        plt.title('Standard Deviation of Images')

        plt.show()
    # Example usage
    image1_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))

    image2_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))


    rmse = calculate_rmse(image1_path, image2_path)
    # plot_standard_deviation(image1_path, image2_path)
    print('RMSE:', rmse)


def pnsr():
    def calculate_psnr(image1_path, image2_path):
        # Open the images
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # Convert the images to grayscale
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        # Convert the images to NumPy arrays
        array1 = np.array(image1)
        array2 = np.array(image2)

        # Calculate the squared error between the images
        squared_error1 = np.square(array1)
        squared_error2 = np.square(array2)

        # Calculate the mean squared error
        mse1 = np.mean(squared_error1)
        mse2 = np.mean(squared_error2)

        # Calculate the maximum possible pixel value
        max_pixel_value1 = np.max(array1)
        max_pixel_value2 = np.max(array2)

        # Calculate the PSNR
        psnr1 = 20 * np.log10(max_pixel_value1) - 10 * np.log10(mse1)
        psnr2 = 20 * np.log10(max_pixel_value2) - 10 * np.log10(mse2)

        return psnr1,psnr2

    # Example usage
    image1_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))

    image2_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))

    psnr = calculate_psnr(image1_path, image2_path)
    print('PSNR:', psnr)

def minkowski_norm():
    def calculate_minkowski_norm(image1_path, image2_path, p):
        # Open the images
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # Convert the images to grayscale
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        # Convert the images to NumPy arrays
        array1 = np.array(image1)
        array2 = np.array(image2)

        # Calculate the absolute difference between the images
        abs_diff = np.abs(array1 - array2)

        # Calculate the Minkowski norm
        norm = np.sum(np.power(abs_diff, p)) ** (1 / p)

        return norm

    # Example usage
    image1_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))

    image2_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))

    p = 2  # Choose the desired value of p (order of the norm)

    minkowski_norm = calculate_minkowski_norm(image1_path, image2_path, p)
    print('Minkowski Norm:', minkowski_norm)


def c_ssim():
    def calculate_ssim(image1_path, image2_path):
        # Open the images
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # Convert the images to grayscale
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        # Convert the images to NumPy arrays
        array1 = np.array(image1)
        array2 = np.array(image2)

        # Calculate the structural similarity
        ssim_score = ssim(array1, array2)

        return ssim_score

    # Example usage
    image1_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))

    image2_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                      filetypes=(("Image Files", "*.jpg;*.jpeg;*.png; *.bmp"), ("All Files", "*.*")))


    ssim_score = calculate_ssim(image1_path, image2_path)
    print('SSIM:', ssim_score)

lf = ttk.LabelFrame(root, text='загрузка изначального изображения')
lf.grid( padx=10, pady=12,ipadx=10, ipady=5, sticky=tk.NW )

lf_b = ttk.LabelFrame(root, text='Геометрическая коррекция')
lf_b.grid(padx=10, pady=10,ipadx=35, ipady=5,sticky=tk.NW )

lf_r = ttk.LabelFrame(root, text='Радиометрическая коррекция')
lf_r.grid( padx=10, pady=10,ipadx=30, ipady=5,sticky=tk.NW )

lf_f = ttk.LabelFrame(root, text='Шумоподавление')
lf_f.grid(padx=10, pady=10,ipadx=38, ipady=5,sticky=tk.NW )

lf_o = ttk.LabelFrame(root, text='Другие операции')
# lf_o.grid(column=2, row= 0, padx=10, pady=10,ipadx=10, ipady=5,sticky=tk.SE)
lf_o.place(x=1000,y=10, relwidth=0.15, relheight=0.2)


lf_c = ttk.LabelFrame(root, text='Оценка качества изображения')
# lf_o.grid(column=2, row= 0, padx=10, pady=10,ipadx=10, ipady=5,sticky=tk.SE)
lf_c.place(x=1000,y=210, relwidth=0.15, relheight=0.2)
#

# # images
lt_img = tk.Label(root)
lt_img.place(x=300,y=40)
img1 = cv2.imread("C:/Users/Chuee/Desktop/diplom/back.jpg")
img1 = cv2.resize(img1, (320, 320))
img1 = Image.fromarray(img1)
photo1 = ImageTk.PhotoImage(image=img1)
lt = tk.Label(root, text=' original img')
lt.place(x=400, y=10)
lt_img.configure(image=photo1)
lt_img.image = photo1

rt_img = tk.Label(root)
rt_img.place(x=650,y=40)
photo2 = ImageTk.PhotoImage(image=img1)
rt = tk.Label(root, text=' original img')
rt.place(x=700, y=10)
rt_img.configure(image=photo1)
rt_img.image = photo1

lb_img = tk.Label(root)
lb_img.place(x=300,y=390)
photo2 = ImageTk.PhotoImage(image=img1)
lb1 = tk.Label(root, text=' changed')
lb1.place( x=400, y=362)
lb_img.configure(image=photo1)
lb_img.image = photo1

rb_img = tk.Label(root)
rb_img.place(x=650,y=390)
photo2 = ImageTk.PhotoImage(image=img1)
rb = tk.Label(root, text=' changed')
rb.place(x=700, y=362)
rb_img.configure(image=photo1)
rb_img.image = photo1

# лин фильтры - средние, лапласовские _ сглаживащие фильтры -
#creat buttons
upload_img= ttk.Button(lf, text = 'загрузить IR изображение',width=25, command = lambda: upload_image())
# upload_img.place(x=10,y=30)
upload_img.grid(ipadx=5, ipady=2)
# upload_img.pack(side=LEFT)

upload_img2= ttk.Button(lf, text = 'загрузить VID изображение',width=25, command = lambda: upload_image1())
upload_img2.grid(ipadx=5, ipady=2)

button1 = ttk.Button(lf_b, text = 'дисторсия на VID изображении',width=25, command = lambda:distors_vid())
button1.grid(ipadx=5, ipady=2)
#
button2 = ttk.Button(lf_b, text = 'дисторсия на ИК',width=25, command = lambda: distors_ik())
button2.grid(ipadx=5, ipady=2)

button3 = ttk.Button(lf_b, text = 'гомография для ИК',width=25, command = lambda: IR_homograh())
button3.grid(ipadx=5, ipady=2)

button4 = ttk.Button(lf_b, text = 'гомография для Vid',width=25, command = lambda: IR_homograh())
button4.grid(ipadx=5, ipady=2)

button5 = ttk.Button(lf_r, text = 'контрастность',width=25, command = lambda: contrast())
button5.grid(padx=3, ipady=2)
f = Entry(lf_r, width=15)
f.grid(padx=3, ipady=2)

button6 = ttk.Button(lf_r, text = 'яркость',width=25, command = lambda: brightness())
button6.grid(padx=3, ipady=2)
br = Entry(lf_r, width=15)
br.grid(padx=3, ipady=2)

button7 = ttk.Button(lf_r, text = 'гамма коррекция',width=25, command = lambda: gamma_korection())
button7.grid(padx=3, ipady=2)

gam = Entry(lf_r, width=15)
gam.grid(padx=3, ipady=2)

button8 = ttk.Button(lf_f, text = 'Гаусс фильтр',width=25, command = lambda: gauss_img())
button8.grid(padx=3, ipady=2)
kern_gauss = Entry(lf_f, width=15)
kern_gauss.grid(padx=3, ipady=2)

button9 = ttk.Button(lf_f, text = 'Средний',width=25, command = lambda: sredniy())
button9.grid(padx=3, ipady=2)
sred = Entry(lf_f, width=15)
sred.grid(padx=3, ipady=2)

button10 = ttk.Button(lf_f, text = 'Медианный',width=25, command = lambda: median())
button10.grid(padx=3, ipady=2)
m = Entry(lf_f, width=15)
m.grid(padx=1, ipady=2)

button11 = ttk.Button(lf_f, text = 'Фильтр лапласа',width=25, command = lambda: laplc_img())
button11.grid(padx=3, ipady=2)
kern = Entry(lf_f, width=15)
kern.grid(padx=3, ipady=2)
#
button14 = ttk.Button(lf_r, text = 'Резкость',width=25, command = lambda: rezkost())
button14.grid(padx=3, ipady=2)
rez = Entry(lf_r, width=15)
rez.grid(padx=3, ipady=2)
#
button15 = ttk.Button(lf_o, text = 'Поиск опорных точек',width=25, command = lambda: points_sift())
button15.grid(padx=3, ipady=2)

button16 = ttk.Button(lf_b, text = 'Поворот изображений',width=25, command = lambda: rotate())
button16.grid(padx=3, ipady=2)
rota = Entry(lf_b, width=15)
rota.grid(padx=1, ipady=2)

button17 = ttk.Button(lf_o, text = 'Комплексирование методом маски',width=25, command = lambda: mast_method())
button17.grid(padx=3, ipady=2)

button18 = ttk.Button(lf_c, text = 'СКО',width=25, command = lambda: rmse())
button18.grid(padx=3, ipady=2)

button19 = ttk.Button(lf_c, text = 'ПСКО',width=25, command = lambda: pnsr())
button19.grid(padx=3, ipady=2)

button20 = ttk.Button(lf_c, text = 'Норма Минковского',width=25, command = lambda: minkowski_norm())
button20.grid(padx=3, ipady=2)

button21 = ttk.Button(lf_c, text = 'Мера структурного подобия',width=25, command = lambda: c_ssim())
button21.grid(padx=3, ipady=2)

root.mainloop()