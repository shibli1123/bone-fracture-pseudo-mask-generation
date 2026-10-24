from __future__ import annotations

import cv2
import numpy as np


def threshold_cam(cam: np.ndarray, mode: str = "global", threshold: float = 0.4) -> np.ndarray:
    cam = np.nan_to_num(np.asarray(cam, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    cam = np.clip(cam, 0.0, 1.0)
    if mode == "global":
        mask = cam >= threshold
    elif mode == "percentile":
        if not 0.0 <= threshold <= 100.0:
            raise ValueError("Percentile threshold must be in [0, 100].")
        mask = cam >= np.percentile(cam, threshold)
    elif mode == "otsu":
        quantized = np.round(cam * 255).astype(np.uint8)
        _, binary = cv2.threshold(quantized, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = binary.astype(bool)
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")
    return mask.astype(np.uint8)


def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number.")
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary
