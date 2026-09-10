"""Detector YOLO (Ultralytics) para maquinaria pesada."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bermguard.types import BBox

_VEHICLE_CLASS_NAMES = {
    "truck",
    "bus",
    "car",
    "train",
}


class YoloVehicleDetector:
    """Wrapper tipado sobre Ultralytics YOLO."""

    def __init__(
        self,
        weights_path: Path,
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.device = _resolve_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.class_names: dict[int, str] = dict(self.model.names)

    def detect(self, frame_bgr: np.ndarray) -> list[BBox]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        r0 = results[0]
        boxes: list[BBox] = []
        if r0.boxes is None:
            return boxes
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            name = self.class_names.get(int(cls_id), str(cls_id))
            area = float((x2 - x1) * (y2 - y1))
            if name not in _VEHICLE_CLASS_NAMES and area < 8000:
                continue
            # COCO no distingue CAEX vs bulldozer; etiqueta genérica honesta.
            label = _map_vehicle_label(name, area)
            boxes.append(
                BBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    conf=float(conf),
                    cls_name=label,
                )
            )
        return boxes


def _map_vehicle_label(coco_name: str, area: float) -> str:
    """Normaliza clases COCO a heavy_vehicle (sin fingir taxonomía minera)."""
    if coco_name in _VEHICLE_CLASS_NAMES or area >= 8000:
        return "heavy_vehicle"
    return "vehicle"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def ensure_weights(weights_path: Path) -> Path:
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    return weights_path
