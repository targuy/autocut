"""
utils.py
Fonctions utilitaires pour gestion de dossiers et extraction de frames.
"""

import cv2
from pathlib import Path

def ensure_dir(path: str) -> None:
    """
    Crée le répertoire si nécessaire.

    :param path: chemin du répertoire à créer
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_frame_at(video_path: str, t: float):
    """
    Extrait la frame à t secondes depuis une vidéo.

    :param video_path: chemin vers le fichier vidéo
    :param t: timestamp en secondes
    :returns: image BGR en numpy.ndarray, ou None si impossible
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None
