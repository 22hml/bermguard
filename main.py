"""BermGuard AI — entrypoint CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bermguard.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "BermGuard AI: detección de maquinaria, segmentación de pretil "
            "y estimación de altura en botaderos mineros."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Carpeta con videos de test (.mp4, .avi, .mov).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Carpeta de salida (se crean subdirectorios por video).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="1",
        choices=["1", "2", "all"],
        help=(
            "Estrategia de pretil (detector vehicular compartido): "
            "1=YOLO-seg, 2=OpenCV clásico, all=ambos."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/yolov8n.pt"),
        help="Pesos YOLO de detección vehicular (compartidos por M1 y M2).",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=None,
        help=(
            "Escala monocular opcional (m/px) para una ROI/escena rectificada. "
            "Si se omite, la altura se reporta en píxeles (sin inventar metros)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo: auto | cpu | cuda | mps.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limitar frames (útil para depuración).",
    )
    parser.add_argument(
        "--berm-seg-weights",
        type=Path,
        default=Path("weights/berm_yolov8n_seg.pt"),
        help="Pesos YOLO-seg del pretil (si no existen, fallback clásico).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"[ERROR] Input no existe: {args.input}", file=sys.stderr)
        return 1

    try:
        run_pipeline(
            input_dir=args.input,
            output_dir=args.output,
            method=args.method,
            berm_seg_weights=args.berm_seg_weights,
            weights_path=args.weights,
            meters_per_pixel=args.meters_per_pixel,
            device=args.device,
            max_frames=args.max_frames,
        )
    except Exception as exc:  # noqa: BLE001 — CLI top-level
        print(f"[ERROR] Falló el pipeline: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
