"""Tracking IoU y semáforo de proximidad."""

from __future__ import annotations

from bermguard.types import AlertLevel, BBox


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._prev: list[BBox] = []

    def update(self, detections: list[BBox]) -> list[BBox]:
        assigned: list[BBox] = []
        used_prev: set[int] = set()

        for det in detections:
            best_iou = 0.0
            best_idx = -1
            for i, prev in enumerate(self._prev):
                if i in used_prev:
                    continue
                score = det.iou(prev)
                if score > best_iou:
                    best_iou = score
                    best_idx = i
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                det.track_id = self._prev[best_idx].track_id
                used_prev.add(best_idx)
            else:
                det.track_id = self._next_id
                self._next_id += 1
            assigned.append(det)

        self._prev = assigned
        return assigned


def proximity_level(
    detections: list[BBox],
    yellow_px: float = 180.0,
    red_px: float = 90.0,
) -> tuple[AlertLevel, float | None, list[str]]:
    if len(detections) < 2:
        return "green", None, []

    # Distancia entre puntos de contacto aprox. con el suelo (bottom-center),
    # no el centro visual del bbox (sesgado por altura aparente).
    min_dist: float | None = None
    pair: tuple[BBox, BBox] | None = None
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            dist = (
                (a.ground_cx - b.ground_cx) ** 2 + (a.ground_cy - b.ground_cy) ** 2
            ) ** 0.5
            if min_dist is None or dist < min_dist:
                min_dist = dist
                pair = (a, b)

    assert min_dist is not None and pair is not None
    alerts: list[str] = []
    if min_dist <= red_px:
        level: AlertLevel = "red"
        alerts.append(
            f"PROXIMIDAD CRÍTICA id={pair[0].track_id}/{pair[1].track_id} "
            f"dist_px={min_dist:.1f}"
        )
    elif min_dist <= yellow_px:
        level = "yellow"
        alerts.append(
            f"PROXIMIDAD PRECAUCIÓN id={pair[0].track_id}/{pair[1].track_id} "
            f"dist_px={min_dist:.1f}"
        )
    else:
        level = "green"
    return level, min_dist, alerts
