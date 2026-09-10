#!/usr/bin/env python3
"""Bootstrap de máscaras de pretil con SAM + cajas/prompts por video.

Flujo (samples brief):
  1) Define ROI aproximada del cordón por video (manual).
  2) SAM genera máscara.
  3) Se convierte a polígono YOLO-seg y se guarda overlay QC.

Revisar `data/berm_seg/qc/` y corregir con annotate_berm.py si hace falta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

# ROI xyxy en coordenadas de imagen original (aprox. del cordón / borde con material).
# Ajustadas tras inspección visual de frames mid de video_01–04.
VIDEO_PROMPTS: dict[str, list[dict]] = {
    "video_01": [
        # Cresta del botadero / cordón horizontal detrás de la flota
        {"bboxes": [[80, 360, 1860, 560]], "points": [[960, 430]], "labels": [1]},
    ],
    "video_02": [
        # Cordón elevado a la izquierda del camino
        {"bboxes": [[0, 280, 420, 700]], "points": [[180, 480]], "labels": [1]},
        # Cresta de material detrás del CAEX
        {"bboxes": [[250, 200, 1050, 360]], "points": [[600, 280]], "labels": [1]},
    ],
    "video_03": [
        # Montículo / cordón inferior izquierdo
        {"bboxes": [[0, 460, 320, 720]], "points": [[120, 600]], "labels": [1]},
        # Ledge / cresta media detrás del camión
        {"bboxes": [[80, 260, 1180, 420]], "points": [[640, 340]], "labels": [1]},
    ],
    "video_04": [
        # Cordones / montículos de material en zona de trabajo
        {"bboxes": [[40, 320, 720, 620]], "points": [[360, 460]], "labels": [1]},
        {"bboxes": [[0, 520, 420, 720]], "points": [[160, 620]], "labels": [1]},
    ],
}


def mask_to_yolo_poly(mask: np.ndarray, min_area: float = 800.0) -> list[float] | None:
    """Máscara binaria -> polígono YOLO normalizado (class omitido)."""
    h, w = mask.shape[:2]
    m = (mask > 0).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    eps = 0.008 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 3:
        return None
    pts = approx.reshape(-1, 2).astype(np.float32)
    xs = np.clip(pts[:, 0] / w, 0, 1)
    ys = np.clip(pts[:, 1] / h, 0, 1)
    out: list[float] = []
    for x, y in zip(xs, ys):
        out.extend([float(x), float(y)])
    return out


def run_sam_on_image(model: SAM, image_bgr: np.ndarray, prompts: list[dict]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    union = np.zeros((h, w), dtype=np.uint8)
    for pr in prompts:
        kwargs = {}
        if "bboxes" in pr:
            kwargs["bboxes"] = pr["bboxes"]
        if "points" in pr:
            kwargs["points"] = pr["points"]
            kwargs["labels"] = pr.get("labels", [1] * len(pr["points"]))
        results = model(image_bgr, verbose=False, **kwargs)
        if not results:
            continue
        r0 = results[0]
        if r0.masks is None:
            continue
        masks = r0.masks.data.cpu().numpy()
        for m in masks:
            mh, mw = m.shape
            if (mh, mw) != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            union = np.maximum(union, (m > 0.5).astype(np.uint8))
    return union


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=Path("data/berm_seg/raw_frames"))
    ap.add_argument("--sam", type=Path, default=Path("weights/sam_b.pt"))
    ap.add_argument("--out-labels", type=Path, default=Path("data/berm_seg/labels_raw"))
    ap.add_argument("--qc", type=Path, default=Path("data/berm_seg/qc"))
    args = ap.parse_args()

    args.out_labels.mkdir(parents=True, exist_ok=True)
    args.qc.mkdir(parents=True, exist_ok=True)

    model = SAM(str(args.sam))
    frames = sorted(args.frames.glob("*.jpg"))
    stats = {"ok": 0, "empty": 0}

    for fp in frames:
        stem = fp.stem  # video_01_f00137
        video_key = "_".join(stem.split("_")[:2])  # video_01
        prompts = VIDEO_PROMPTS.get(video_key)
        if not prompts:
            print(f"[SKIP] sin prompts: {fp.name}")
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        mask = run_sam_on_image(model, img, prompts)
        polys = []
        # separar componentes para múltiples instancias
        nlab, labels = cv2.connectedComponents(mask)
        for lab in range(1, nlab):
            comp = (labels == lab).astype(np.uint8)
            poly = mask_to_yolo_poly(comp)
            if poly is not None:
                polys.append(poly)

        label_path = args.out_labels / f"{fp.stem}.txt"
        if polys:
            lines = ["0 " + " ".join(f"{v:.6f}" for v in poly) for poly in polys]
            label_path.write_text("\n".join(lines) + "\n")
            stats["ok"] += 1
        else:
            label_path.write_text("")
            stats["empty"] += 1

        # QC overlay
        qc = img.copy()
        color = (0, 0, 255)
        overlay = qc.copy()
        overlay[mask > 0] = (0, 40, 200)
        cv2.addWeighted(overlay, 0.45, qc, 0.55, 0, qc)
        for poly in polys:
            h, w = img.shape[:2]
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            cv2.polylines(qc, [pts.astype(np.int32)], True, color, 2)
        cv2.putText(
            qc,
            f"{fp.name} | polys={len(polys)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(str(args.qc / fp.name), qc)
        print(f"[OK] {fp.name} polys={len(polys)} area={int(mask.sum())}")

    meta = {"stats": stats, "prompts": VIDEO_PROMPTS}
    (args.out_labels.parent / "bootstrap_meta.json").write_text(json.dumps(meta, indent=2))
    print("done", stats)


if __name__ == "__main__":
    main()
