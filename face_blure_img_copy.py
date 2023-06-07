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

# Apply blur on entire image
blurred_image = cv2.GaussianBlur(image_copy, (27, 27), 0)

if landmarks.size > 0:
    final_faces = np.zeros((height, width, 3), np.uint8)
    for single_face in landmarks:
        convexhull = cv2.convexHull(single_face)

        # 2 Face extraction
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask ,[convexhull], True, 255, 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_extracted = cv2.bitwise_and(image_copy, image_copy, mask = mask)

        # Replace the face region with original face in final_faces
        final_faces = cv2.bitwise_or(final_faces, face_extracted)

    # Extract background which is actually the whole blurred image
    result = cv2.bitwise_or(blurred_image, final_faces)

else:
    result = blurred_image

image_name = image_name.split('.')[0] + "_blur.jpg"
cv2.imwrite(image_name, result)
