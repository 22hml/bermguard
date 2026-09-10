"""Estimación de pretil con YOLO-seg entrenado en máscaras de cordón."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from bermguard.types import BBox, BermEstimate

_MODEL: YOLO | None = None
_MODEL_PATH: str | None = None


def _load_model(weights: Path | str) -> YOLO:
    global _MODEL, _MODEL_PATH
    path = str(weights)
    if _MODEL is None or _MODEL_PATH != path:
        _MODEL = YOLO(path)
        _MODEL_PATH = path
    return _MODEL


def _subtract_vehicles(mask: np.ndarray, detections: list[BBox] | None, pad: float = 0.06) -> np.ndarray:
    if not detections:
        return mask
    out = mask.copy()
    h, w = out.shape
    for d in detections:
        px = pad * max(d.width, 1.0)
        py = pad * max(d.height, 1.0)
        x1, x2 = max(0, int(d.x1 - px)), min(w, int(d.x2 + px))
        y1, y2 = max(0, int(d.y1 - py)), min(h, int(d.y2 + py))
        out[y1:y2, x1:x2] = 0
    return out


def _polylines_from_mask(
    mask: np.ndarray, step: int = 8
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    h, w = mask.shape
    xs, crest, ground = [], [], []
    for x in range(0, w, step):
        col = mask[:, x] > 0
        if not np.any(col):
            continue
        ys = np.where(col)[0]
        xs.append(float(x))
        crest.append(float(ys.min()))
        ground.append(float(ys.max()))
    if len(xs) < 4:
        return [], []
    return list(zip(xs, crest)), list(zip(xs, ground))


def estimate_berm_seg(
    frame_bgr: np.ndarray,
    weights: Path | str,
    *,
    meters_per_pixel: float | None = None,
    scale_valid: bool = False,
    conf: float = 0.15,
    device: str | None = None,
    detections: list[BBox] | None = None,
) -> BermEstimate:
    """Segmenta cordón/pretil. Sin máscara confiable → unknown (no inventa)."""
    model = _load_model(weights)
    kwargs = {"verbose": False, "conf": conf}
    if device:
        kwargs["device"] = device
    results = model.predict(frame_bgr, **kwargs)
    if not results or results[0].masks is None or len(results[0].masks) == 0:
        return BermEstimate(status="unknown", reason="yolo-seg: sin máscara de pretil", confidence=0.0)

    r0 = results[0]
    masks = r0.masks.data.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy() if r0.boxes is not None else np.ones(len(masks))
    h, w = frame_bgr.shape[:2]

    scored = []
    for i, m in enumerate(masks):
        mm = m
        if mm.shape != (h, w):
            mm = cv2.resize(mm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (mm > 0.5).astype(np.uint8)
        binary = _subtract_vehicles(binary, detections)
        area = int(binary.sum())
        if area < max(200, 0.0008 * h * w):
            continue
        ys = np.where(binary > 0)[0]
        xs = np.where(binary > 0)[1]
        if ys.size == 0:
            continue
        cy = float(np.median(ys))
        # Rechazar cerro/horizonte lejano (pretil = cordón bajo en plataforma)
        if cy < 0.38 * h:
            continue
        crest_tmp = float(ys.min())
        ground_tmp = float(ys.max())
        height_tmp = ground_tmp - crest_tmp
        span_tmp = float(xs.max() - xs.min()) / max(1.0, float(w))
        # Preferir cordones bajos y alargados (evitar acopios / muros altos)
        if height_tmp > 0.14 * h:
            continue
        if height_tmp < 4:
            continue
        c = float(confs[i]) if i < len(confs) else 0.5
        thickness = max(1.0, height_tmp / h)
        score = c * (0.4 + 0.6 * min(1.0, span_tmp / 0.25)) / (0.25 + thickness * 2.0)
        scored.append((score, c, binary, height_tmp, span_tmp))

    if not scored:
        return BermEstimate(
            status="unknown",
            reason="yolo-seg: sin cordón bajo creíble",
            confidence=0.0,
        )

    scored.sort(key=lambda t: t[0], reverse=True)
    conf_score, binary = scored[0][1], scored[0][2]

    # Unir otros cordones bajos cercanos en Y (misma plataforma)
    for _, _, bb, ht, sp in scored[1:]:
        if ht > 0.18 * h:
            continue
        ys = np.where(binary > 0)[0]
        ys2 = np.where(bb > 0)[0]
        if ys.size and ys2.size and abs(float(np.median(ys)) - float(np.median(ys2))) < 0.12 * h:
            binary = np.maximum(binary, bb)

    crest_pl, ground_pl = _polylines_from_mask(binary, step=max(4, w // 80))
    if len(crest_pl) < 4 or len(ground_pl) < 4:
        return BermEstimate(
            status="unknown",
            reason="yolo-seg: no se pudo derivar cresta/pie",
            confidence=conf_score,
            mask=binary,
        )

    crest_y = float(np.median([y for _, y in crest_pl]))
    ground_y = float(np.median([y for _, y in ground_pl]))
    height_px = max(0.0, ground_y - crest_y)
    # Cordón bajo/fino: techo relativo evita pintar muros altos como pretil
    if height_px < 4 or height_px > 0.14 * h:
        return BermEstimate(
            status="unknown",
            reason="yolo-seg: altura px fuera de rango de cordón bajo",
            confidence=conf_score,
            mask=binary,
        )

    xs = [x for x, _ in crest_pl]
    span = (max(xs) - min(xs)) / max(1.0, float(w))
    # Los pretiles pequeños pueden ser cortos; no exigir span enorme
    if span < 0.04:
        return BermEstimate(
            status="unknown",
            reason="yolo-seg: span horizontal bajo",
            confidence=conf_score,
            mask=binary,
        )

    height_m = None
    mpp = meters_per_pixel if scale_valid else None
    if scale_valid and mpp is not None and mpp > 0:
        height_m = height_px * mpp

    return BermEstimate(
        status="detected",
        crest_y=crest_y,
        ground_y=ground_y,
        height_px=height_px,
        height_m=height_m,
        meters_per_pixel=mpp,
        scale_valid=bool(scale_valid and height_m is not None),
        confidence=min(1.0, conf_score * (0.6 + 0.4 * min(1.0, span / 0.5))),
        reason="yolo-seg",
        mask=binary,
        crest_polyline=crest_pl,
        ground_polyline=ground_pl,
    )


def default_berm_weights() -> Path:
    return Path("weights/berm_yolov8n_seg.pt")
