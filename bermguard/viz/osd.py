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

_PRETIL_FILL = (40, 40, 200)
_PRETIL_EDGE = (30, 30, 255)
_EDGE_COLOR = (220, 180, 40)


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

    # Máscara YOLO-seg (si existe)
    if berm.mask is not None and berm.status == "detected":
        try:
            m = np.asarray(berm.mask)
            if m.ndim == 2 and m.shape[0] == h and m.shape[1] == w:
                overlay = frame.copy()
                overlay[m > 0] = _PRETIL_FILL
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        except Exception:
            pass

    # Borde de botadero (orientación) — nunca se etiqueta como pretil solo
    if berm.edge_polyline and len(berm.edge_polyline) >= 2:
        epts = np.array([[int(round(x)), int(round(y))] for x, y in berm.edge_polyline], dtype=np.int32)
        cv2.polylines(frame, [epts], False, _EDGE_COLOR, 2, cv2.LINE_AA)
        x0, y0 = int(epts[0, 0]), int(epts[0, 1])
        cv2.putText(
            frame,
            "borde botadero",
            (max(8, x0), max(40, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _EDGE_COLOR,
            1,
            cv2.LINE_AA,
        )

    if berm.status != "detected":
        msg = "PRETIL: SIN DETECCIÓN CONFIABLE"
        if berm.status == "edge_only":
            msg = "PRETIL: SOLO BORDE (sin cordón)"
        cv2.rectangle(frame, (0, 36), (frame.shape[1], 68), (30, 30, 30), -1)
        cv2.putText(
            frame,
            msg,
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        if berm.reason:
            cv2.putText(
                frame,
                berm.reason[:80],
                (12, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
        return

    crest = berm.crest_polyline or []
    ground = berm.ground_polyline or []
    if len(crest) < 2 or len(ground) < 2:
        return

    crest_pts = np.array([[int(round(x)), int(round(y))] for x, y in crest], dtype=np.int32)
    ground_pts = np.array([[int(round(x)), int(round(y))] for x, y in ground], dtype=np.int32)
    poly = np.vstack([crest_pts, ground_pts[::-1]])
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], _PRETIL_FILL)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
    cv2.polylines(frame, [crest_pts], False, _PRETIL_EDGE, 4, cv2.LINE_AA)
    cv2.polylines(frame, [ground_pts], False, (50, 200, 50), 2, cv2.LINE_AA)

    mid_i = len(crest) // 2
    mx, my = int(crest[mid_i][0]), int(crest[mid_i][1])
    gy = int(ground[min(mid_i, len(ground) - 1)][1])
    cv2.arrowedLine(frame, (mx, gy), (mx, my), (255, 255, 255), 2, tipLength=0.05)

    if berm.scale_valid and berm.height_m is not None:
        label = f"Pretil ~ {berm.height_m:.2f} m ({berm.height_px:.0f}px)"
    elif berm.height_px is not None:
        label = f"Pretil ~ {berm.height_px:.0f}px (sin calibración métrica)"
    else:
        label = "Pretil detectado"
    cv2.putText(
        frame,
        label,
        (min(w - 420, max(10, mx + 8)), max(28, my - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
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
    # Proximidad (vehículos) — independiente del estado del pretil
    color = _LEVEL_COLOR[result.proximity_level]
    text = f"PROXIMIDAD: {result.proximity_level.upper()}"
    if result.proximity_distance_px is not None:
        text += f" | dist={result.proximity_distance_px:.0f}px"
    text += f" | t={result.timestamp_s:.1f}s"

    berm = result.berm
    if berm is None or berm.status != "detected":
        # No presentar el frame como "todo OK" si el pretil es desconocido
        berm_txt = " | PRETIL: DESCONOCIDO"
        if berm is not None and berm.status == "edge_only":
            berm_txt = " | PRETIL: SOLO BORDE"
        text += berm_txt
        # Banner ámbar si pretil no confirmado (aunque proximidad sea green)
        banner_color = (0, 180, 255)
    else:
        text += f" | PRETIL: OK conf={berm.confidence:.2f}"
        banner_color = color

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        banner_color if berm is None or berm.status != "detected" else color,
        2,
        cv2.LINE_AA,
    )
