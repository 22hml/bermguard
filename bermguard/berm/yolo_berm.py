"""Estimación de pretil asistida por detecciones (método 1)."""

from __future__ import annotations

from pathlib import Path

from bermguard.berm.classical import estimate_berm_classical
from bermguard.berm.seg_yolo import default_berm_weights, estimate_berm_seg
from bermguard.types import BBox, BermEstimate


def estimate_berm_method1(
    frame_bgr,
    detections: list[BBox],
    meters_per_pixel: float | None = None,
    *,
    scale_valid: bool = False,
    berm_seg_weights: Path | str | None = None,
    device: str | None = None,
) -> BermEstimate:
    """Método 1: vehículos YOLO + pretil por segmentación si hay pesos; si no, clásico.

    Las cajas enmascaran oclusiones; **no** son rasante global.
    """
    weights = Path(berm_seg_weights) if berm_seg_weights else default_berm_weights()
    if weights.exists():
        return estimate_berm_seg(
            frame_bgr,
            weights,
            meters_per_pixel=meters_per_pixel,
            scale_valid=scale_valid,
            device=device,
            detections=detections,
        )

    return estimate_berm_classical(
        frame_bgr,
        meters_per_pixel=meters_per_pixel,
        detections=detections,
        scale_valid=scale_valid,
    )
