"""Escala monocular y altura de pretil."""

from __future__ import annotations

from bermguard.types import BBox

DEFAULT_TIRE_DIAMETER_M = 4.0
DEFAULT_BERM_GUIDELINE_RATIO = 0.50


def estimate_meters_per_pixel(
    detections: list[BBox],
    frame_height: int,
    override: float | None = None,
) -> float:
    if override is not None and override > 0:
        return override

    truck_like = [
        d for d in detections if d.cls_name in {"heavy_vehicle", "truck", "vehicle"}
    ]
    if truck_like:
        tallest = max(truck_like, key=lambda d: d.height)
        if tallest.height > 10:
            tire_px = 0.30 * tallest.height
            return DEFAULT_TIRE_DIAMETER_M / tire_px

    return 15.0 / max(frame_height, 1)


def guideline_min_berm_m(tire_diameter_m: float = DEFAULT_TIRE_DIAMETER_M) -> float:
    return tire_diameter_m * DEFAULT_BERM_GUIDELINE_RATIO
