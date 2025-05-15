
import numpy as np
from ultralytics import YOLO
import torch
import cv2


class SkinSegmenter:
    def __init__(self, weights: str, device: str = "cuda:0", area_thresh: float = 0.01):
        self.device = device
        self.model = YOLO(weights, task="segment")
        self.area_thresh = area_thresh  # seuil de rejet des petits masques

    def percentage(self, image: np.ndarray, person_mask: np.ndarray = None) -> float:
        # Appliquer la segmentation
        results = self.model(image, device=self.device)[0]

        # Vérifier les masques
        if not hasattr(results, "masks") or results.masks is None or len(results.masks.data) == 0:
            return 0.0

        h, w = image.shape[:2]
        total_area = h * w

        # Masques binaires
        masks = results.masks.data.cpu().numpy()  # shape: (N, H, W)
        combined_mask = np.zeros((h, w), dtype=bool)

        for m in masks:
            if np.sum(m) / total_area < self.area_thresh:
                continue
            combined_mask |= (m > 0.5)

        # Appliquer un masque de personne si fourni
        if person_mask is not None:
            combined_mask &= (person_mask > 0)

        skin_area = np.sum(combined_mask)
        return 100.0 * skin_area / (np.sum(person_mask) if person_mask is not None else total_area)

    def debug_overlay(self, image: np.ndarray) -> np.ndarray:
        # Génère une image superposée pour debug visuel
        results = self.model(image, device=self.device)[0]
        if not hasattr(results, "masks") or results.masks is None or len(results.masks.data) == 0:
            return image.copy()

        masks = results.masks.data.cpu().numpy()
        combined_mask = np.any(masks > 0.5, axis=0)
        overlay = image.copy()
        overlay[combined_mask] = [0, 255, 0]  # vert = peau
        return overlay
