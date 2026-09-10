#!/usr/bin/env python3
"""Anotador interactivo de polígonos YOLO-seg para pretil.

Uso:
  python scripts/annotate_berm.py --frames data/berm_seg/raw_frames \\
      --labels data/berm_seg/labels_raw

Controles:
  click izq  = agregar vértice
  click der  = cerrar polígono actual
  u          = deshacer último vértice
  n / espacio = siguiente imagen (guarda)
  p          = anterior
  c          = borrar anotaciones de esta imagen
  q / esc    = salir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_polys(path: Path, w: int, h: int) -> list[np.ndarray]:
    if not path.exists() or not path.read_text().strip():
        return []
    polys = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        vals = list(map(float, parts[1:]))
        pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        polys.append(pts)
    return polys


def save_polys(path: Path, polys: list[np.ndarray], w: int, h: int) -> None:
    lines = []
    for pts in polys:
        if len(pts) < 3:
            continue
        norm = []
        for x, y in pts:
            norm.extend([max(0.0, min(1.0, float(x) / w)), max(0.0, min(1.0, float(y) / h))])
        lines.append("0 " + " ".join(f"{v:.6f}" for v in norm))
    path.write_text(("\n".join(lines) + "\n") if lines else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=Path("data/berm_seg/raw_frames"))
    ap.add_argument("--labels", type=Path, default=Path("data/berm_seg/labels_raw"))
    args = ap.parse_args()
    args.labels.mkdir(parents=True, exist_ok=True)

    frames = sorted(args.frames.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"Sin frames en {args.frames}")

    idx = 0
    current: list[tuple[float, float]] = []
    win = "BermGuard annotate (berm polygon)"

    def redraw() -> np.ndarray:
        img = cv2.imread(str(frames[idx]))
        h, w = img.shape[:2]
        polys = load_polys(args.labels / f"{frames[idx].stem}.txt", w, h)
        vis = img.copy()
        for pts in polys:
            cv2.fillPoly(vis, [pts.astype(np.int32)], (40, 40, 180))
            cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.addWeighted(vis, 0.45, img, 0.55, 0, vis)
        if current:
            pts = np.array(current, dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 255), 2)
            for x, y in current:
                cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 255), -1)
        cv2.putText(
            vis,
            f"{idx+1}/{len(frames)} {frames[idx].name} | click=vertice | RMB=cerrar | n=next | c=clear",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return vis

    def on_mouse(event, x, y, flags, param):  # noqa: ARG001
        nonlocal current
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and len(current) >= 3:
            img = cv2.imread(str(frames[idx]))
            h, w = img.shape[:2]
            lp = args.labels / f"{frames[idx].stem}.txt"
            polys = load_polys(lp, w, h)
            polys.append(np.array(current, dtype=np.float32))
            save_polys(lp, polys, w, h)
            current = []

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = redraw()
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("n"), 32):
            current = []
            idx = min(len(frames) - 1, idx + 1)
        if key == ord("p"):
            current = []
            idx = max(0, idx - 1)
        if key == ord("u") and current:
            current.pop()
        if key == ord("c"):
            (args.labels / f"{frames[idx].stem}.txt").write_text("")
            current = []

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
