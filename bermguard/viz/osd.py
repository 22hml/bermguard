"""Overlay OSD sobre frames."""

from __future__ import annotations

import cv2
import numpy as np

from bermguard.types import AlertLevel, BBox, BermEstimate, FrameResult

_LEVEL_COLOR = {
    "green": (80, 200, 80),
    "yellow": (0, 220, 255),
    "red": (40, 40, 240),
}


def draw_osd(frame_bgr: np.ndarray, result: FrameResult) -> np.ndarray:
    out = frame_bgr.copy()
    if result.berm is not None:
        _draw_berm(out, result.berm)
    for det in result.detections:
        _draw_bbox(out, det, result.proximity_level)
    _draw_banner(out, result)
    return out


def _draw_berm(frame: np.ndarray, berm: BermEstimate) -> None:
    h, w = frame.shape[:2]
    if isinstance(berm.mask, np.ndarray) and berm.mask.shape[:2] == (h, w):
        color = np.zeros_like(frame)
        color[:, :] = (180, 120, 40)  # ámbar minería
        alpha = (berm.mask.astype(np.float32) / 255.0) * 0.35
        alpha = alpha[:, :, None]
        frame[:] = (frame.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).astype(
            np.uint8
        )

    cy = int(berm.crest_y)
    gy = int(berm.ground_y)
    cv2.line(frame, (0, cy), (w, cy), (0, 255, 255), 2)
    cv2.line(frame, (0, gy), (w, gy), (0, 180, 0), 2)
    mid_x = w // 2
    cv2.arrowedLine(frame, (mid_x, gy), (mid_x, cy), (255, 255, 255), 2, tipLength=0.03)
    cv2.putText(
        frame,
        f"Pretil ~ {berm.height_m:.2f} m ({berm.height_px:.0f}px)",
        (mid_x + 10, max(30, cy - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_bbox(frame: np.ndarray, det: BBox, level: AlertLevel) -> None:
    color = _LEVEL_COLOR[level] if level != "green" else (255, 180, 60)
    p1 = (int(det.x1), int(det.y1))
    p2 = (int(det.x2), int(det.y2))
    cv2.rectangle(frame, p1, p2, color, 2)
    label = f"{det.cls_name}"
    if det.track_id is not None:
        label += f" #{det.track_id}"
    label += f" {det.conf:.2f}"
    cv2.putText(
        frame,
        label,
        (p1[0], max(20, p1[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_banner(frame: np.ndarray, result: FrameResult) -> None:
    color = _LEVEL_COLOR[result.proximity_level]
    text = f"PROXIMIDAD: {result.proximity_level.upper()}"
    if result.proximity_distance_px is not None:
        text += f" | dist={result.proximity_distance_px:.0f}px"
    text += f" | t={result.timestamp_s:.1f}s"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
