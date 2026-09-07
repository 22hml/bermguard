"""Segmentación / estimación de pretil — método clásico (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from bermguard.types import BermEstimate


def estimate_berm_classical(
    frame_bgr: np.ndarray,
    meters_per_pixel: float,
    roi_top_ratio: float = 0.40,
    roi_bottom_ratio: float = 0.92,
) -> BermEstimate:
    """Estima cresta y rasante del pretil con bordes + morfología.

    Se limita a una franja horizontal del tercio inferior (botadero),
    buscando un *ridge* compacto (pretil) en lugar de toda la ladera.
    """
    h, w = frame_bgr.shape[:2]
    y0 = int(h * roi_top_ratio)
    y1 = int(h * roi_bottom_ratio)
    roi = frame_bgr[y0:y1, :]
    rh = roi.shape[0]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Gradiente vertical: el pretil suele ser un salto de intensidad horizontal
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(sobel_y)
    mag = cv2.GaussianBlur(mag, (1, 9), 0)

    # Densidad por fila (promedio en banda central, evita bordes de frame)
    x0, x1 = w // 6, w - w // 6
    row_score = mag[:, x0:x1].mean(axis=1)

    if float(row_score.max()) < 1e-3:
        crest_local = int(0.35 * rh)
        ground_local = int(min(rh - 1, crest_local + 0.12 * rh))
        conf = 0.1
    else:
        smooth = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 11), 0).ravel()
        crest_local = int(np.argmax(smooth))

        # Rasante: poco después de la cresta, donde el gradiente cae (suelo / pie)
        search_from = min(rh - 2, crest_local + max(3, int(0.02 * rh)))
        search_to = min(rh - 1, crest_local + max(8, int(0.22 * rh)))
        segment = smooth[search_from:search_to]
        if len(segment) == 0:
            ground_local = min(rh - 1, crest_local + int(0.1 * rh))
        else:
            # pie ≈ mínimo local tras la cresta, o percentil bajo
            ground_local = search_from + int(np.argmin(segment))
        conf = float(min(1.0, smooth.max() / (smooth.mean() + 1e-6) / 8.0))

    # Altura de pretil típica: acotar a banda razonable en px (~0.5–3.5 m)
    height_px_raw = float(max(1, ground_local - crest_local))
    max_px = max(8.0, 3.5 / max(meters_per_pixel, 1e-6))
    min_px = max(3.0, 0.4 / max(meters_per_pixel, 1e-6))
    height_px = float(np.clip(height_px_raw, min_px, max_px))
    ground_local = int(crest_local + height_px)

    crest_y = float(y0 + crest_local)
    ground_y = float(y0 + ground_local)
    height_m = height_px * meters_per_pixel

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(crest_y) : int(ground_y), x0:x1] = 255

    return BermEstimate(
        crest_y=crest_y,
        ground_y=ground_y,
        height_px=height_px,
        height_m=height_m,
        meters_per_pixel=meters_per_pixel,
        confidence=conf,
        mask=mask,
    )
