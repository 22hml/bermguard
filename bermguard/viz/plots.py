"""Gráficos de altura y distribución espacial."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from bermguard.types import FrameResult


def plot_berm_height(frames: list[FrameResult], out_path: Path) -> None:
    times = [f.timestamp_s for f in frames if f.berm is not None]
    heights = [f.berm.height_m for f in frames if f.berm is not None]  # type: ignore[union-attr]
    fig, ax = plt.subplots(figsize=(10, 4))
    if times:
        ax.plot(times, heights, color="#f59e0b", linewidth=2, label="Altura pretil")
        ax.fill_between(times, heights, alpha=0.15, color="#f59e0b")
        ax.axhline(
            np.mean(heights),
            color="#38bdf8",
            linestyle="--",
            label=f"Media={np.mean(heights):.2f} m",
        )
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura estimada (m)")
    ax.set_title("Evolución de altura de pretil vs tiempo")
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
