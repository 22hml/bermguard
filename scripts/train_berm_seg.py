#!/usr/bin/env python3
"""Arma dataset YOLO-seg (train/val) y entrena yolov8n-seg para pretil."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


def build_dataset(frames: Path, labels: Path, dataset: Path, val_ratio: float, seed: int) -> Path:
    imgs = sorted(frames.glob("*.jpg"))
    labeled = []
    for im in imgs:
        lp = labels / f"{im.stem}.txt"
        if lp.exists() and lp.read_text().strip():
            labeled.append(im)
    if len(labeled) < 4:
        raise SystemExit(f"Se necesitan >=4 imágenes anotadas; hay {len(labeled)}")

    random.Random(seed).shuffle(labeled)
    n_val = max(1, int(round(len(labeled) * val_ratio)))
    val_set = set(labeled[:n_val])
    train_set = [p for p in labeled if p not in val_set]

    if dataset.exists():
        shutil.rmtree(dataset)
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)

    def copy_split(paths: list[Path], split: str) -> None:
        for im in paths:
            shutil.copy2(im, dataset / "images" / split / im.name)
            shutil.copy2(labels / f"{im.stem}.txt", dataset / "labels" / split / f"{im.stem}.txt")

    copy_split(train_set, "train")
    copy_split(list(val_set), "val")

    yaml_path = dataset / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: berm",
                "",
            ]
        )
    )
    print(f"dataset: train={len(train_set)} val={len(val_set)} -> {yaml_path}")
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=Path("data/berm_seg/raw_frames"))
    ap.add_argument("--labels", type=Path, default=Path("data/berm_seg/labels_raw"))
    ap.add_argument("--dataset", type=Path, default=Path("data/berm_seg/dataset"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--out-weights", type=Path, default=Path("weights/berm_yolov8n_seg.pt"))
    args = ap.parse_args()

    yaml_path = build_dataset(args.frames, args.labels, args.dataset, args.val_ratio, seed=42)

    model = YOLO("yolov8n-seg.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/berm_seg",
        name="train",
        exist_ok=True,
        patience=25,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=8.0,
        translate=0.08,
        scale=0.4,
        fliplr=0.5,
        mosaic=0.8,
        close_mosaic=15,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out_weights)
    print(f"[OK] pesos -> {args.out_weights}")


if __name__ == "__main__":
    main()
