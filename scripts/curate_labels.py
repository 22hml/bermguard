#!/usr/bin/env python3
"""Genera etiquetas YOLO-seg de pretil (banda cresta–base) por video de samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def poly_from_norm(norm_pts: list[tuple[float, float]], w: int, h: int) -> np.ndarray:
    return np.array([[x * w, y * h] for x, y in norm_pts], dtype=np.float32)


def save_polys(path: Path, polys: list[np.ndarray], w: int, h: int) -> None:
    lines = []
    for pts in polys:
        if len(pts) < 3:
            continue
        flat = []
        for x, y in pts:
            flat.extend([float(np.clip(x / w, 0, 1)), float(np.clip(y / h, 0, 1))])
        lines.append("0 " + " ".join(f"{v:.6f}" for v in flat))
    path.write_text(("\n".join(lines) + "\n") if lines else "")


def berm_video01(w: int, h: int) -> list[np.ndarray]:
    """Cordones bajos en plataforma (video_01)."""
    a = poly_from_norm(
        [
            (0.18, 0.50),
            (0.28, 0.48),
            (0.38, 0.49),
            (0.40, 0.54),
            (0.30, 0.56),
            (0.20, 0.55),
        ],
        w,
        h,
    )
    b = poly_from_norm(
        [
            (0.36, 0.51),
            (0.48, 0.49),
            (0.58, 0.50),
            (0.60, 0.55),
            (0.50, 0.57),
            (0.38, 0.56),
        ],
        w,
        h,
    )
    c = poly_from_norm(
        [
            (0.55, 0.52),
            (0.68, 0.50),
            (0.78, 0.52),
            (0.78, 0.57),
            (0.68, 0.58),
            (0.55, 0.57),
        ],
        w,
        h,
    )
    return [a, b, c]


def berm_video02(w: int, h: int) -> list[np.ndarray]:
    """Banda fina cresta–base (video_02)."""
    return [
        poly_from_norm(
            [
                (0.04, 0.485),
                (0.22, 0.465),
                (0.40, 0.470),
                (0.60, 0.485),
                (0.80, 0.505),
                (0.96, 0.525),
                (0.96, 0.575),
                (0.80, 0.560),
                (0.60, 0.545),
                (0.40, 0.535),
                (0.22, 0.530),
                (0.04, 0.545),
            ],
            w,
            h,
        )
    ]


def berm_video03(w: int, h: int) -> list[np.ndarray]:
    """Banda fina cresta–base (video_03)."""
    return [
        poly_from_norm(
            [
                (0.02, 0.48),
                (0.25, 0.46),
                (0.45, 0.47),
                (0.65, 0.49),
                (0.85, 0.51),
                (0.98, 0.53),
                (0.98, 0.60),
                (0.85, 0.58),
                (0.65, 0.56),
                (0.45, 0.55),
                (0.25, 0.54),
                (0.02, 0.55),
            ],
            w,
            h,
        )
    ]


def berm_video04(w: int, h: int) -> list[np.ndarray]:
    """Banda fina cresta–base nocturna (video_04)."""
    return [
        poly_from_norm(
            [
                (0.05, 0.50),
                (0.25, 0.48),
                (0.45, 0.49),
                (0.65, 0.51),
                (0.85, 0.53),
                (0.98, 0.55),
                (0.98, 0.62),
                (0.85, 0.60),
                (0.65, 0.58),
                (0.45, 0.57),
                (0.25, 0.56),
                (0.05, 0.57),
            ],
            w,
            h,
        )
    ]


MANUAL = {
    "video_01": berm_video01,
    "video_02": berm_video02,
    "video_03": berm_video03,
    "video_04": berm_video04,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Curar etiquetas YOLO-seg de pretil")
    ap.add_argument("--frames", type=Path, default=Path("data/berm_seg/raw_frames"))
    ap.add_argument("--out-labels", type=Path, default=Path("data/berm_seg/labels"))
    ap.add_argument("--qc", type=Path, default=Path("data/berm_seg/qc"))
    args = ap.parse_args()
    args.out_labels.mkdir(parents=True, exist_ok=True)
    args.qc.mkdir(parents=True, exist_ok=True)

    for fp in sorted(args.frames.glob("*.jpg")):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        video = "_".join(fp.stem.split("_")[:2])
        fn = MANUAL.get(video)
        if fn is None:
            continue
        polys = fn(w, h)
        save_polys(args.out_labels / f"{fp.stem}.txt", polys, w, h)

        qc = img.copy()
        overlay = qc.copy()
        for p in polys:
            cv2.fillPoly(overlay, [p.astype(np.int32)], (0, 40, 200))
            cv2.polylines(qc, [p.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.addWeighted(overlay, 0.45, qc, 0.55, 0, qc)
        cv2.putText(
            qc,
            f"{fp.name} berm n={len(polys)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(str(args.qc / fp.name), qc)

    print(f"labels -> {args.out_labels} | qc -> {args.qc}")


if __name__ == "__main__":
    main()
