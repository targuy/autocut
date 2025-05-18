"""
detectors/mask.py
Calcul du pourcentage d'occlusion du visage (masquage) par seuillage de l'image.
"""
import numpy as np
import cv2

def face_mask_percentage(face_roi: np.ndarray) -> float:
    """
    Calcule le pourcentage de pixels masqués (occlus) dans une image de visage.
    :param face_roi: image BGR du visage (numpy.ndarray)
    :returns: pourcentage de pixels masqués (0.0 à 100.0)
    """
    if face_roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # Seuillage inverse : pixels sombres considérés comme masqués
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    masked = np.count_nonzero(mask)
    total = face_roi.shape[0] * face_roi.shape[1]
    if total == 0:
        return 0.0
    return 100.0 * masked / total
