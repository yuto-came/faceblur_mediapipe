import cv2
import mediapipe as mp
import numpy as np
from facial_landmarks import FaceLandmarks

# Load face landmarks
fl = FaceLandmarks()

image_name = "yuto_test.png"
image = cv2.imread(image_name)

image_copy = image.copy()
height, width, _ = image.shape

# 1 Face landmark detection
landmarks = fl.get_facial_landmarks(image)

if landmarks.size > 0:
    face_extracted_list = []
    image_copy = cv2.GaussianBlur(image_copy, (27, 27), 0) 
    mask = np.zeros((height, width), np.uint8)
    for single_face in landmarks:
        convexhull = cv2.convexHull(single_face)

        # 2 Face blurring
        cv2.polylines(mask ,[convexhull], True, 255, 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        # Extract the Face
        face_extracted = cv2.bitwise_and(image_copy, image_copy, mask = mask)
        face_extracted_list.append(face_extracted)
        # blureed_image = cv2.GaussianBlur(face_extracted,(27, 27), 0)

    # Extract background
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(image, image, mask= background_mask)
    for face in face_extracted_list:
        # Final result
        result = cv2.add(background, face)

else:
    result = image_copy

image_name = image_name.split('.')[0] + "_blur.jpg"
cv2.imwrite(image_name, result)