import cv2 as cv
import mediapipe as mp
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

# Load face landmarks
fl = FaceLandmarks()

cap = cv.VideoCapture("person_walking.mp4")

while True:
  
  ret, frame = cap.read()
  frame = cv.resize(frame, None, fx = 0.5, fy = 0.5)
  frame_copy = frame.copy()
  height, width, _ = frame.shape

  # 1 Face landmark detection
  landmarks = fl.get_facial_landmarks(frame)
  convexhull = cv.convexHull(landmarks)

  # 2 Face blurring
  mask = np.zeros((height, width), np.uint8)
  cv.polylines(mask ,[convexhull], True, 255, 3)
  cv.fillConvexPoly(mask, convexhull, 255)

  # Extract the Face
  frame_copy = cv.blur(frame_copy, (27, 27))  
  face_extracted = cv.bitwise_and(frame_copy, frame_copy, mask = mask)
  # blureed_frame = cv.GaussianBlur(face_extracted,(27, 27), 0)

  # Extract background
  background_mask = cv.bitwise_not(mask)
  background = cv.bitwise_and(frame, frame, mask= background_mask)

  # Final result
  result = cv.add(background, face_extracted)

  cv.imshow('Result', result)
  cv.imshow('Frame', frame) 
  key = cv.waitKey(30)
  if key == 27:
    break
cap.release()
cv.distroyAllWindows()