import os

# Set the directory path to the folder containing the images
directory = "/Users/yukaaoyama/oedk-fastener-sorter/darknet/build/darknet/x64/data/obj"

# Get a list of all the file names in the directory
file_names = os.listdir(directory)

# Filter the file names to only include image files
image_file_names = [f for f in file_names if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

# Print the list of image file names
for image_file in image_file_names:
    file = open('/Users/yukaaoyama/oedk-fastener-sorter/darknet/build/darknet/x64/data/train.txt', "a")
    file.write(str(image_file)+ os.linesep)
    file.close()