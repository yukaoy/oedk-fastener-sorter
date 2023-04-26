import cv2
import numpy as np
import os

# set the path to your input images and annotations
input_path = 'darknet/build/darknet/x64/data/obj/'

# create a new directory to save the resized images and annotations
output_path = 'darknet/build/darknet/x64/data/obj/'
# if not os.path.exists(output_path):
#     os.makedirs(output_path)

# set the maximum size for your output images
max_size = 416

# loop over all the images in the input directory
for filename in os.listdir(input_path):

    # Check if the file is a JPG image
    if not filename.endswith('.jpg'):
        continue

    # read the image
    img = cv2.imread(os.path.join(input_path, filename))
    # Convert image from BGR to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    
    # calculate the aspect ratio of the original image
    aspect_ratio = width / height

    # resize the image to the maximum desired size, while preserving the aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(new_height * aspect_ratio)
    
    resized_img = cv2.resize(img, (new_width, new_height))
    
    # create a new blank image with the desired size, and fill the added padding with black
    padded_img = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    padding_top = int((max_size - new_height) / 2)
    padding_bottom = max_size - new_height - padding_top
    padding_left = int((max_size - new_width) / 2)
    padding_right = max_size - new_width - padding_left

    padded_img[padding_top:padding_top+new_height, padding_left:padding_left+new_width, :] = resized_img
    
    # save the resized image to the output directory
    cv2.imwrite(os.path.join(output_path, filename), padded_img)

    # modify the corresponding annotation file
    txt_filename = filename + '.txt'
    with open(os.path.join(input_path, txt_filename), 'r') as f:
        lines = f.readlines()
    with open(os.path.join(output_path, txt_filename), 'w') as f:
        for line in lines:
            line = line.strip().split()
            x_center = float(line[1])
            y_center = float(line[2])
            w = float(line[3])
            h = float(line[4])
            # modify the bounding box coordinates according to the resizing and padding
            x_center = x_center * new_width / width + padding_left
            y_center = y_center * new_height / height + padding_top
            w = w * new_width / width
            h = h * new_height / height
            # write the modified annotation line to the new file
            f.write(f"{line[0]} {x_center/max_size:.6f} {y_center/max_size:.6f} {w/max_size:.6f} {h/max_size:.6f}\n")
