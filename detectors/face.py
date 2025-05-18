"""
detectors/face.py
Détection de visages dans une image ROI via Ultralytics YOLO.
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple

BBox = Tuple[int, int, int, int, float]  # (x, y, w, h, confidence)

class FaceDetector:
    """
    Détecte les visages dans une image.
    :param weights: chemin vers le fichier .pt du modèle YOLO de détection de visage
    :param device: 'cuda' ou 'cpu'
    :param conf_thres: seuil de confiance minimal
    """
    def __init__(self, weights: str, device: str = "cuda", conf_thres: float = 0.0) -> None:
        # Charger le modèle YOLO de détection faciale
        self.model = YOLO(weights)
        self.device = device
        self.conf_thres = conf_thres

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Exécute la détection de visages sur l'image.
        :param frame: image BGR sous forme de numpy.ndarray
        :returns: liste de tuples (x, y, w, h, confidence) pour chaque visage détecté
        """
        results = self.model(frame, device=self.device)[0]
        rects = []
        for box, conf in zip(results.boxes.xyxy.cpu().numpy(),
                             results.boxes.conf.cpu().numpy()):
            if conf >= self.conf_thres:
                x1, y1, x2, y2 = box.astype(int)
                rects.append((x1, y1, x2 - x1, y2 - y1, float(conf)))
        return rects
