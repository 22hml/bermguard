#!/usr/bin/env python3
"""Extrae fotogramas de videos para anotar / entrenar segmentación de pretil."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser(description="Extraer frames para dataset de pretil")
    p.add_argument("--input", type=Path, default=Path("data/samples"))
    p.add_argument("--output", type=Path, default=Path("data/berm_seg/raw_frames"))
    p.add_argument("--per-video", type=int, default=10)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for vp in sorted(args.input.glob("video_0*.mp4")):
        cap = cv2.VideoCapture(str(vp))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        w, h = int(cap.get(3)), int(cap.get(4))
        idxs = [int(n * (i + 1) / (args.per_video + 1)) for i in range(args.per_video)]
        saved = 0
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"{vp.stem}_f{fi:05d}.jpg"
            cv2.imwrite(str(args.output / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            manifest.append(
                {"file": name, "video": vp.name, "frame": fi, "w": w, "h": h, "t": fi / fps}
            )
            saved += 1
        cap.release()
        print(f"[OK] {vp.name}: {saved} frames ({w}x{h})")

    (args.output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Total {len(manifest)} -> {args.output}")


if __name__ == "__main__":
    main()
