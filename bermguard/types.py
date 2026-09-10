"""Tipos compartidos del pipeline BermGuard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AlertLevel = Literal["green", "yellow", "red"]
BermStatus = Literal["detected", "edge_only", "unknown"]


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_name: str
    track_id: int | None = None

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union > 0 else 0.0


@dataclass
class BermEstimate:
    """Estimación de pretil / borde en un frame.

    ``status``:
      - detected: cordón con cresta y pie independientes
      - edge_only: solo borde de botadero (no equivale a pretil)
      - unknown: sin evidencia confiable (no inventar geometría)
    """

    status: BermStatus = "unknown"
    crest_y: float | None = None
    ground_y: float | None = None
    height_px: float | None = None
    height_m: float | None = None
    meters_per_pixel: float | None = None
    scale_valid: bool = False
    confidence: float = 0.0
    reason: str = ""
    mask: object | None = None
    crest_polyline: list[tuple[float, float]] = field(default_factory=list)
    ground_polyline: list[tuple[float, float]] = field(default_factory=list)
    edge_polyline: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class FrameResult:
    frame_idx: int
    timestamp_s: float
    detections: list[BBox] = field(default_factory=list)
    berm: BermEstimate | None = None
    proximity_level: AlertLevel = "green"
    proximity_distance_px: float | None = None
    alerts: list[str] = field(default_factory=list)


@dataclass
class VideoStats:
    video_name: str
    method: str
    fps_input: float
    frames_processed: int
    wall_time_s: float
    avg_infer_fps: float
    alert_count: int
    mean_berm_height_m: float | None
    device: str
    berm_detect_rate: float = 0.0
