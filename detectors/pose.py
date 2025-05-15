# detectors/pose.py
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# 3D model points of facial landmarks (nose tip, chin, eye corners, mouth)
model_points = np.array([
    [0.0, 0.0, 0.0],           # Nose tip
    [0.0, -63.6, -12.5],       # Chin
    [-43.3, 32.7, -26.0],      # Left eye left corner
    [43.3, 32.7, -26.0],       # Right eye right corner
    [-28.9, -28.9, -24.1],     # Left mouth corner
    [28.9, -28.9, -24.1]       # Right mouth corner
])

landmark_indices = [1, 152, 263, 33, 287, 57]

def estimate_head_pose(image, face_box):
    """
    Estime l’orientation de la tête (pitch, yaw, roll) à partir d’un crop visage.
    `face_box` = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = face_box
    face = image[y1:y2, x1:x2]

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    results = mp_face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return None

    # Option simplifiée : estimer orientation par approximation
    landmarks = results.multi_face_landmarks[0].landmark
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # Dummy approximation (à remplacer par solvePnP si besoin)
    dx = right_eye.x - left_eye.x
    dy = nose.y - (left_eye.y + right_eye.y) / 2
    pitch = dy * 100
    yaw = dx * 100
    roll = 0.0  # non calculé ici

    return pitch, yaw, roll
