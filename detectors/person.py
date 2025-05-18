"""
detectors/person.py
Détection de personnes via Ultralytics YOLO.
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple

BBox = Tuple[int, int, int, int, float]  # (x, y, w, h, confidence)

class PersonDetector:
    """
    Détecteur de personnes.
    :param weights: chemin vers le .pt Ultralytics (segmentation personnes)
    :param device: "cuda" ou "cpu"
    :param conf_thres: confiance minimale pour accepter une détection
    """
    def __init__(self, weights: str, device: str = "cuda", conf_thres: float = 0.0) -> None:
        self.model = YOLO(weights)
        self.device = device
        self.conf_thres = conf_thres

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Détecte les personnes dans l'image.
        :param frame: image BGR en numpy.ndarray
        :returns: liste de tuples (x, y, w, h, confidence) par personne détectée
        """
        results = self.model(frame, device=self.device)[0]
        boxes = []
        for box, conf in zip(results.boxes.xyxy.cpu().numpy(),
                             results.boxes.conf.cpu().numpy()):
            if conf >= self.conf_thres:
                x1, y1, x2, y2 = box.astype(int)
                boxes.append((x1, y1, x2 - x1, y2 - y1, float(conf)))
        return boxes
