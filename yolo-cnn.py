import cv2
import numpy as np
import torch

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Typically, YOLO models expect input images of size (416, 416).
    resized_frame = cv2.resize(frame, (416, 416))
    # YOLO models expect input images in RGB format.
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Normalize the pixel values of the image. YOLO models usually expect input images with pixel values scaled to the range [0, 1].
    normalized_frame = rgb_frame / 255.0
    # Expand the dimensions of the image to match the input shape of the YOLO model. The YOLO model expects a 4D tensor of shape (batch_size, height, width, channels) as input.
    input_image = np.expand_dims(normalized_frame, axis=0)
    frame = torch.from_numpy(frame)

    # Pass the preprocessed image into the YOLO model
    # model_output = yolo_model(frame)