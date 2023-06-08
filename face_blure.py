import cv2
import mediapipe as mp
import numpy as np

class FaceLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                                max_num_faces=100,
                                                refine_landmarks=False,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)

    def get_facial_landmarks(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

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

# Load face landmarks
fl = FaceLandmarks()

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    frame_copy = frame.copy()
    height, width, _ = frame.shape

    # 1 Face landmark detection
    landmarks = fl.get_facial_landmarks(frame)

    if landmarks.size > 0:
        face_extracted_list = []
        frame_copy = cv2.GaussianBlur(frame_copy, (41, 41), 0) 
        mask = np.zeros((height, width), np.uint8)
        for single_face in landmarks:
            convexhull = cv2.convexHull(single_face)

            # 2 Face blurring
            cv2.polylines(mask ,[convexhull], True, 255, 3)
            cv2.fillConvexPoly(mask, convexhull, 255)

            # Extract the Face
            face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask = mask)
            face_extracted_list.append(face_extracted)
            # blureed_image = cv2.GaussianBlur(face_extracted,(27, 27), 0)

        # Extract background
        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask= background_mask)
        for face in face_extracted_list:
            # Final result
            result = cv2.add(background, face)

    else:
        result = frame_copy

    cv2.imshow('Result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.distroyAllWindows()
