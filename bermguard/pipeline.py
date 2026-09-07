"""Orquestación del pipeline BermGuard."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from bermguard.berm import estimate_berm_classical, estimate_berm_method1
from bermguard.detectors import YoloVehicleDetector, ensure_weights
from bermguard.geometry import estimate_meters_per_pixel, guideline_min_berm_m
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

    for video_path in videos:
        for m in methods:
            _process_video(
                video_path=video_path,
                output_dir=output_dir,
                method=m,
                detector=detector,
                meters_per_pixel=meters_per_pixel,
                max_frames=max_frames,
                device=device if detector is None else detector.device,
            )


def _process_video(
    video_path: Path,
    output_dir: Path,
    method: str,
    detector: YoloVehicleDetector | None,
    meters_per_pixel: float | None,
    max_frames: int | None,
    device: str,
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
    frames_meta: list[FrameResult] = []
    alerts_all: list[dict[str, object]] = []
    mpp_running: float | None = meters_per_pixel

    t0 = time.perf_counter()
    idx = 0
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
        # Método 2: sin detector deep; solo pretil clásico (benchmark)

        mpp = estimate_meters_per_pixel(
            detections=detections,
            frame_height=height,
            override=mpp_running if mpp_running is not None else meters_per_pixel,
        )
        # Congelar escala: preferir primera observación con vehículos; si no, primer frame
        if mpp_running is None:
            if detections or idx == 0:
                mpp_running = mpp
        if mpp_running is not None:
            mpp = mpp_running

        if method == "1":
            berm = estimate_berm_method1(enhanced, detections, meters_per_pixel=mpp)
        else:
            berm = estimate_berm_classical(enhanced, meters_per_pixel=mpp)

        level, dist_px, alerts = proximity_level(detections)
        guide = guideline_min_berm_m()
        if berm.height_m < guide * 0.75:
            alerts.append(
                f"PRETIL BAJO: {berm.height_m:.2f} m < 75% guía ({guide:.2f} m)"
            )

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
        # Etiqueta de método / lighting
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

    heights = [f.berm.height_m for f in frames_meta if f.berm is not None]
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
    )
    metadata = {
        "video": video_path.name,
        "method": method,
        "device": device,
        "fps_input": stats.fps_input,
        "frames_processed": stats.frames_processed,
        "wall_time_s": round(stats.wall_time_s, 3),
        "avg_throughput_fps": round(stats.avg_infer_fps, 3),
        "mean_berm_height_m": (
            None
            if stats.mean_berm_height_m is None
            else round(stats.mean_berm_height_m, 3)
        ),
        "guideline_min_berm_m": guideline_min_berm_m(),
        "meters_per_pixel": mpp_running,
        "alert_count": stats.alert_count,
        "alerts": alerts_all[:500],  # cap para no inflar
        "artifacts": {
            "osd_video": writer_path.name,
            "berm_height_plot": "berm_height_vs_time.png",
            "spatial_plot": "vehicle_spatial_distribution.png",
        },
    }
    with open(out_sub / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(
        f"[OK] {video_path.name} method={method} -> {out_sub} "
        f"({idx} frames, {stats.avg_infer_fps:.2f} FPS eff.)"
    )


def iter_videos(input_dir: Path) -> Iterable[Path]:
    for p in sorted(Path(input_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p
