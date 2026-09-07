"""Preprocesado robusto a cambios de iluminación."""

from __future__ import annotations

import cv2
import numpy as np


def enhance_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """CLAHE sobre canal L (LAB) para día/noche/polvo."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def estimate_lighting_regime(frame_bgr: np.ndarray) -> str:
    """Clasificación gruesa: day | dusk | night."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    if mean_val < 55:
        return "night"
    if mean_val < 110:
        return "dusk"
    return "day"
