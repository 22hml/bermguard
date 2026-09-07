"""Segmentación / estimación de pretil — método clásico (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from bermguard.types import BermEstimate


def estimate_berm_classical(
    frame_bgr: np.ndarray,
    meters_per_pixel: float,
    roi_top_ratio: float = 0.35,
) -> BermEstimate:
    """Estima cresta y rasante del pretil con bordes + morfología."""
    h, w = frame_bgr.shape[:2]
    y0 = int(h * roi_top_ratio)
    roi = frame_bgr[y0:h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    row_density = edges.mean(axis=1).astype(np.float64)
    if row_density.max() < 1e-6:
        crest_local = int(0.35 * roi.shape[0])
        ground_local = int(0.85 * roi.shape[0])
        conf = 0.15
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        smooth = cv2.GaussianBlur(row_density.reshape(-1, 1), (1, 15), 0).ravel()
        threshold = 0.45 * float(smooth.max())
        candidates = np.where(smooth >= threshold)[0]
        crest_local = int(candidates[0]) if len(candidates) else int(np.argmax(smooth))

        lower = smooth[int(0.55 * len(smooth)) :]
        if len(lower) == 0:
            ground_local = roi.shape[0] - 1
        else:
            ground_local = int(0.55 * len(smooth) + int(np.argmin(lower)))
        ground_local = max(ground_local, crest_local + 5)
        conf = float(min(1.0, smooth.max() / 80.0))

        mask = np.zeros((h, w), dtype=np.uint8)
        crest_y_abs = y0 + crest_local
        ground_y_abs = y0 + ground_local
        mask[crest_y_abs:ground_y_abs, :] = 255
        mask[:, : w // 8] = 0
        mask[:, -w // 8 :] = 0

    crest_y = float(y0 + crest_local)
    ground_y = float(y0 + ground_local)
    height_px = max(0.0, ground_y - crest_y)
    height_m = height_px * meters_per_pixel

    return BermEstimate(
        crest_y=crest_y,
        ground_y=ground_y,
        height_px=height_px,
        height_m=height_m,
        meters_per_pixel=meters_per_pixel,
        confidence=conf,
        mask=mask,
    )
