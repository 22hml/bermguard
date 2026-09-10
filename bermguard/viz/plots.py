"""Gráficos de altura y distribución espacial."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from bermguard.types import FrameResult


def plot_berm_height(frames: list[FrameResult], out_path: Path) -> None:
    times_m: list[float] = []
    heights_m: list[float] = []
    times_px: list[float] = []
    heights_px: list[float] = []
    for f in frames:
        if f.berm is None or f.berm.status != "detected":
            continue
        if f.berm.height_m is not None:
            times_m.append(f.timestamp_s)
            heights_m.append(f.berm.height_m)
        elif f.berm.height_px is not None:
            times_px.append(f.timestamp_s)
            heights_px.append(f.berm.height_px)

    fig, ax = plt.subplots(figsize=(10, 4))
    if times_m:
        ax.plot(times_m, heights_m, color="#f59e0b", linewidth=2, label="Altura pretil (m)")
        ax.fill_between(times_m, heights_m, alpha=0.15, color="#f59e0b")
        ax.axhline(
            float(np.mean(heights_m)),
            color="#38bdf8",
            linestyle="--",
            label=f"Media={np.mean(heights_m):.2f} m",
        )
        ax.set_ylabel("Altura estimada (m)")
        ax.set_title("Evolución de altura de pretil vs tiempo (calibrado)")
    elif times_px:
        ax.plot(times_px, heights_px, color="#f59e0b", linewidth=2, label="Altura pretil (px)")
        ax.fill_between(times_px, heights_px, alpha=0.15, color="#f59e0b")
        ax.axhline(
            float(np.mean(heights_px)),
            color="#38bdf8",
            linestyle="--",
            label=f"Media={np.mean(heights_px):.0f} px",
        )
        ax.set_ylabel("Altura estimada (px)")
        ax.set_title("Evolución de altura de pretil vs tiempo (sin calibración métrica)")
    else:
        ax.text(
            0.5,
            0.5,
            "Sin detecciones confiables de pretil\nen este segmento",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Evolución de altura de pretil vs tiempo")
        ax.set_ylabel("Altura")

    ax.set_xlabel("Tiempo (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_spatial_distribution(frames: list[FrameResult], out_path: Path) -> None:
    xs: list[float] = []
    ys: list[float] = []
    ts: list[float] = []
    for fr in frames:
        for det in fr.detections:
            xs.append(det.cx)
            ys.append(det.cy)
            ts.append(fr.timestamp_s)

    fig, ax = plt.subplots(figsize=(8, 6))
    if xs:
        sc = ax.scatter(xs, ys, c=ts, cmap="viridis", s=12, alpha=0.75)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Tiempo (s)")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title("Distribución espacial de vehículos en el tiempo")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
