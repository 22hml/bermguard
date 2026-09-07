"""Estimación de pretil asistida por YOLO / ROI vehicular (método 1)."""

from __future__ import annotations

import cv2
import numpy as np

from bermguard.berm.classical import estimate_berm_classical
from bermguard.types import BBox, BermEstimate


def estimate_berm_method1(
    frame_bgr: np.ndarray,
    detections: list[BBox],
    meters_per_pixel: float,
) -> BermEstimate:
    """Método 1: clásico + anclaje opcional a la base de vehículos detectados.

    Si hay detecciones, la rasante se regulariza con el percentil inferior
    de las bases de bbox (suelo operativo bajo CAEX/bulldozer).
    """
    base = estimate_berm_classical(frame_bgr, meters_per_pixel=meters_per_pixel)
    if not detections:
        return base

    bottoms = [d.y2 for d in detections]
    ground_from_vehicles = float(np.percentile(bottoms, 75))
    # Mezcla: más peso al ancla vehicular si confiable
    ground_y = 0.55 * base.ground_y + 0.45 * ground_from_vehicles
    ground_y = max(ground_y, base.crest_y + 5.0)
    height_px = ground_y - base.crest_y
    height_m = height_px * meters_per_pixel

    mask = base.mask
    if isinstance(mask, np.ndarray):
        h, w = mask.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cy = int(base.crest_y)
        gy = int(ground_y)
        mask[cy:gy, w // 8 : -w // 8] = 255
        # Contorno suave
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)

    return BermEstimate(
        crest_y=base.crest_y,
        ground_y=ground_y,
        height_px=height_px,
        height_m=height_m,
        meters_per_pixel=meters_per_pixel,
        confidence=min(1.0, base.confidence + 0.15),
        mask=mask,
    )
