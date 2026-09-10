"""Orquestación del pipeline BermGuard."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from bermguard.berm import estimate_berm_classical, estimate_berm_method1
from bermguard.berm.temporal import BermTemporalFilter
from bermguard.detectors import YoloVehicleDetector, ensure_weights
from bermguard.geometry import guideline_min_berm_m
from bermguard.preprocess import enhance_frame, estimate_lighting_regime
from bermguard.tracking import IoUTracker, proximity_level
from bermguard.types import FrameResult, VideoStats
from bermguard.viz import draw_osd, plot_berm_height, plot_spatial_distribution

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    method: str,
    weights_path: Path,
    meters_per_pixel: float | None = None,
    device: str = "auto",
    max_frames: int | None = None,
    berm_seg_weights: Path | None = None,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        raise FileNotFoundError(f"No se encontraron videos en {input_dir}")

    methods: list[str]
    if method == "all":
        methods = ["1", "2"]
    else:
        methods = [method]

    detector: YoloVehicleDetector | None = None
    if "1" in methods:
        wpath = ensure_weights(weights_path)
        detector = YoloVehicleDetector(weights_path=wpath, device=device)

    # Solo calibración explícita publica metros
    scale_valid = meters_per_pixel is not None and meters_per_pixel > 0

    for video_path in videos:
        for m in methods:
            _process_video(
                video_path=video_path,
                output_dir=output_dir,
                method=m,
                detector=detector,
                meters_per_pixel=meters_per_pixel,
                scale_valid=scale_valid,
                max_frames=max_frames,
                device=device if detector is None else detector.device,
                berm_seg_weights=berm_seg_weights,
            )


def _process_video(
    video_path: Path,
    output_dir: Path,
    method: str,
    detector: YoloVehicleDetector | None,
    meters_per_pixel: float | None,
    scale_valid: bool,
    max_frames: int | None,
    device: str,
    berm_seg_weights: Path | None = None,
) -> None:
    stem = video_path.stem
    out_sub = output_dir / f"{stem}_method{method}"
    out_sub.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer_path = out_sub / f"{stem}_osd.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(writer_path), fourcc, fps, (width, height))

    tracker = IoUTracker()
    berm_filter = BermTemporalFilter(confirm=2, clear_after=2)
    frames_meta: list[FrameResult] = []
    alerts_all: list[dict[str, object]] = []

    t0 = time.perf_counter()
    idx = 0
    detect_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and idx >= max_frames:
            break

        enhanced = enhance_frame(frame)
        lighting = estimate_lighting_regime(frame)
        timestamp = idx / fps if fps > 0 else float(idx)

        detections = []
        if method == "1":
            assert detector is not None
            detections = detector.detect(enhanced)
            detections = tracker.update(detections)

        # Escala solo para uso métrico si hay calibración explícita
        mpp_for_metric = meters_per_pixel if scale_valid else None

        if method == "1":
            berm = estimate_berm_method1(
                enhanced,
                detections,
                meters_per_pixel=mpp_for_metric,
                scale_valid=scale_valid,
                berm_seg_weights=berm_seg_weights,
                device=device,
            )
        else:
            berm = estimate_berm_classical(
                enhanced,
                meters_per_pixel=mpp_for_metric,
                scale_valid=scale_valid,
            )
        berm = berm_filter.update(berm)
        if berm.status == "detected":
            detect_count += 1

        level, dist_px, alerts = proximity_level(detections)

        if berm.status != "detected":
            alerts.append(
                "PRETIL: SIN DETECCIÓN CONFIABLE"
                if berm.status == "unknown"
                else "PRETIL: SOLO BORDE (sin cordón verificado)"
            )
        elif berm.scale_valid and berm.height_m is not None:
            guide = guideline_min_berm_m()
            if berm.height_m < guide * 0.75:
                alerts.append(
                    f"PRETIL BAJO: {berm.height_m:.2f} m < 75% guía ({guide:.2f} m)"
                )
        elif berm.height_px is not None:
            # Alerta relativa en px solo informativa; no se vende como metrología
            alerts.append(f"PRETIL DETECTADO (~{berm.height_px:.0f}px, sin calibración m)")

        result = FrameResult(
            frame_idx=idx,
            timestamp_s=timestamp,
            detections=detections,
            berm=berm,
            proximity_level=level,
            proximity_distance_px=dist_px,
            alerts=alerts,
        )
        for a in alerts:
            alerts_all.append(
                {
                    "frame": idx,
                    "time_s": timestamp,
                    "lighting": lighting,
                    "alert": a,
                }
            )

        osd = draw_osd(frame, result)
        cv2.putText(
            osd,
            f"method={method} | light={lighting}",
            (12, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        writer.write(osd)
        frames_meta.append(result)
        idx += 1

    cap.release()
    writer.release()
    wall = time.perf_counter() - t0

    plot_berm_height(frames_meta, out_sub / "berm_height_vs_time.png")
    plot_spatial_distribution(frames_meta, out_sub / "vehicle_spatial_distribution.png")

    heights = [
        f.berm.height_m
        for f in frames_meta
        if f.berm is not None and f.berm.status == "detected" and f.berm.height_m is not None
    ]
    heights_px = [
        f.berm.height_px
        for f in frames_meta
        if f.berm is not None and f.berm.status == "detected" and f.berm.height_px is not None
    ]
    detect_rate = detect_count / idx if idx else 0.0
    stats = VideoStats(
        video_name=video_path.name,
        method=method,
        fps_input=fps,
        frames_processed=idx,
        wall_time_s=wall,
        avg_infer_fps=(idx / wall) if wall > 0 else 0.0,
        alert_count=len(alerts_all),
        mean_berm_height_m=float(np.mean(heights)) if heights else None,
        device=device,
        berm_detect_rate=detect_rate,
    )
    metadata = {
        "video": video_path.name,
        "method": method,
        "device": device,
        "resolution": {"width": width, "height": height},
        "fps_input": stats.fps_input,
        "frames_processed": stats.frames_processed,
        "wall_time_s": round(stats.wall_time_s, 3),
        "avg_throughput_fps": round(stats.avg_infer_fps, 3),
        "berm_detect_rate": round(detect_rate, 3),
        "mean_berm_height_m": (
            None if stats.mean_berm_height_m is None else round(stats.mean_berm_height_m, 3)
        ),
        "mean_berm_height_px": (
            None if not heights_px else round(float(np.mean(heights_px)), 1)
        ),
        "scale_calibrated": scale_valid,
        "guideline_min_berm_m": guideline_min_berm_m() if scale_valid else None,
        "meters_per_pixel": meters_per_pixel if scale_valid else None,
        "alert_count": stats.alert_count,
        "alerts": alerts_all[:500],
        "artifacts": {
            "osd_video": writer_path.name,
            "berm_height_plot": "berm_height_vs_time.png",
            "spatial_plot": "vehicle_spatial_distribution.png",
        },
        "notes": (
            "height_m solo si --meters-per-pixel explícito; "
            "pretil vía YOLO-seg si weights/berm_yolov8n_seg.pt existe; "
            "unknown no inventa polígono."
        ),
        "berm_seg_weights": (
            str(berm_seg_weights)
            if berm_seg_weights is not None
            else str(Path("weights/berm_yolov8n_seg.pt"))
        ),
    }
    with open(out_sub / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(
        f"[OK] {video_path.name} method={method} -> {out_sub} "
        f"({idx} frames, {stats.avg_infer_fps:.2f} FPS, detect={detect_rate:.0%})"
    )


def iter_videos(input_dir: Path) -> Iterable[Path]:
    for p in sorted(Path(input_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p
