# detectors/nsfw.py
"""
Wrapper pour NSFWImageDetector
"""

from nsfw_image_detector import NSFWDetector
from PIL import Image
import torch
import numpy as np

class NSFWWrapper:
    def __init__(self, device: str = "cuda:0", dtype: str = "bfloat16"):
        self.device = device
        dtype_attr = getattr(torch, dtype) if hasattr(torch, dtype) else None
        self.detector = NSFWDetector(dtype=dtype_attr, device=device)

    def predict(self, image_np: np.ndarray, mode: str = "high"):
        # Convert BGR (OpenCV) to PIL RGB
        img = Image.fromarray(image_np[..., ::-1])
        is_nsfw = self.detector.is_nsfw(img, mode)
        probas = self.detector.predict_proba(img)
        return is_nsfw, probas
