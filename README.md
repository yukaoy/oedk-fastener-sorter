# oedk-fastener-sorter

main.py -> the original code using object detection (openCV)
yolo-images -> identical code to main.py just for image captures and making annotation .txt files. I change the fastener variable manually before I start capturing images.
preprocessing.py -> correcting bounding box values in annotation .txt files to adjust to padding added in images. Paddings are added in images to adjust to what yolov4 takes in (416x416). The code for image resize is also in this file currently commented out.
copy-images-names.py -> to get images' names into a files used by darknet (train.txt and test.txt)