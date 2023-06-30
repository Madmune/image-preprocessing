from PIL import Image
import os

def invert_images(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over each file in the folder
    for file_name in file_list:
        # Check if the file is an image
        if file_name.endswith(".bmp"):
            # Open the image using PIL
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)

            # Invert the image colors
            inverted_image = Image.eval(image, lambda x: 255 - x)

            # Save the inverted image
            inverted_image_path = os.path.join(folder_path, "inverted_" + file_name)
            inverted_image.save(inverted_image_path)

            print(f"Inverted {file_name} and saved as {inverted_image_path}")

# Specify the folder path where the images are located
folder_path = "C:/Users/Chuee/Desktop/diplom/3"

# Call the function to invert the images in the folder
invert_images(folder_path)