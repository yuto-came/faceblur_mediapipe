# https://pysource.com/blur-faces-in-real-time-with-opencv-mediapipe-and-python
import mediapipe as mp
import cv2
import numpy as np


class FaceLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                                max_num_faces=3,
                                                refine_landmarks=False,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)

    def get_facial_landmarks(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)
        # print("result:", result.multi_face_landmarks)
        print("result qnt:", len(result.multi_face_landmarks))

        if result.multi_face_landmarks:
            facelandmarks_all = []
            for facial_landmarks in result.multi_face_landmarks:
                facelandmarks = []
                for i in range(0, 468):
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    facelandmarks.append([x, y])
                facelandmarks_all.append(facelandmarks)
            return np.array(facelandmarks_all, np.int32)
        
        else:
            return np.array([], np.int32)