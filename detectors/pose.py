# detectors/pose.py
"""
detectors/pose.py
Estimation de l'orientation de la tête (pitch, yaw, roll) via Mediapipe.
"""
try:
    import mediapipe as mp
except ImportError:
    mp = None
import cv2
import numpy as np

# Points 3D de référence (nez, menton, coins des yeux, commissures des lèvres)
model_points = np.array([
    [0.0, 0.0, 0.0],           # Nose tip
    [0.0, -63.6, -12.5],       # Chin
    [-43.3, 32.7, -26.0],      # Left eye left corner
    [43.3, 32.7, -26.0],       # Right eye right corner
    [-28.9, -28.9, -24.1],     # Left mouth corner
    [28.9, -28.9, -24.1]       # Right mouth corner
])
landmark_indices = [1, 152, 263, 33, 287, 57]  # Indices Mediapipe des points correspondants

def estimate_head_pose(image: np.ndarray, face_box: tuple):
    """
    Estime l'orientation de la tête (pitch, yaw, roll) à partir d'un rectangle facial.
    `face_box` = (x1, y1, x2, y2) coordonnées du cadre de visage.
    Retourne (pitch, yaw, roll) en degrés, ou None si estimation échoue.
    """
    # Si Mediapipe n'est pas disponible, retourner orientation neutre (0,0,0)
    if mp is None:
        return (0.0, 0.0, 0.0)
    x1, y1, x2, y2 = face_box
    face = image[y1:y2, x1:x2]
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    results = face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    # Approche simplifiée : calcul d'une approximation d'orientation
    landmarks = results.multi_face_landmarks[0].landmark
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    # Approximation de l'orientation (pitch, yaw basés sur positions normalisées)
    dx = right_eye.x - left_eye.x
    dy = nose.y - (left_eye.y + right_eye.y) / 2
    pitch = dy * 100  # échelle arbitraire
    yaw = dx * 100
    roll = 0.0  # Non calculé dans cette approximation
    return pitch, yaw, roll
