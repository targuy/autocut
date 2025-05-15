"""
classifiers/gender.py
Classification homme/femme sur une image de visage.
"""

import torch
import numpy as np
import cv2
from typing import Tuple

class GenderClassifier:
    """
    Classifie une ROI visage en 'homme' ou 'femme'.

    :param weights: chemin vers le fichier .pt du classifieur
    :param device: 'cuda' ou 'cpu'
    :param img_size: taille de l'image d'entrée (côté carré)
    """
    def __init__(self,
                 weights: str,
                 device: str = "cuda",
                 img_size: int = 128) -> None:
        self.device = device
        self.img_size = img_size
        self.model = torch.load(weights, map_location=device).eval()
        self.labels = ['homme', 'femme']

    def predict(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Prédit le genre de la ROI.

        :param roi: image BGR sous forme de numpy.ndarray
        :returns: (label, confiance)
        """
        img = cv2.resize(roi, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = (torch.from_numpy(img)
                    .permute(2, 0, 1)
                    .float()
                    .div(255)
                    .unsqueeze(0)
                    .to(self.device))
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])
