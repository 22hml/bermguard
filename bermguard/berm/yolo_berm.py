"""Estimación de pretil asistida por detecciones (método 1)."""

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
    """Método 1: clásico + refinamiento suave con bases de bbox cercanas al pretil."""
    base = estimate_berm_classical(frame_bgr, meters_per_pixel=meters_per_pixel)
    if not detections:
        return base

    # Solo anclar si la base del vehículo cae cerca de la rasante clásica
    near: list[float] = []
    for d in detections:
        if abs(d.y2 - base.ground_y) < 0.12 * frame_bgr.shape[0]:
            near.append(d.y2)
    if not near:
        return base

    ground_from_vehicles = float(np.median(near))
    ground_y = 0.7 * base.ground_y + 0.3 * ground_from_vehicles
    ground_y = max(ground_y, base.crest_y + 3.0)
    height_px = ground_y - base.crest_y
    # Mantener cota física razonable
    max_px = max(8.0, 3.5 / max(meters_per_pixel, 1e-6))
    height_px = float(min(height_px, max_px))
    ground_y = base.crest_y + height_px
    height_m = height_px * meters_per_pixel

    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    h, w = mask.shape
    mask[int(base.crest_y) : int(ground_y), w // 6 : -w // 6] = 255
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2)

    return BermEstimate(
        crest_y=base.crest_y,
        ground_y=ground_y,
        height_px=height_px,
        height_m=height_m,
        meters_per_pixel=meters_per_pixel,
        confidence=min(1.0, base.confidence + 0.1),
        mask=mask,
    )
