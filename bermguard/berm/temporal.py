"""Filtro temporal de detecciones de pretil."""

from __future__ import annotations

from bermguard.types import BermEstimate


class BermTemporalFilter:
    """Confirma detecciones y evita arrastrar falsos positivos.

    - Requiere ``confirm`` frames detected seguidos para publicar geometría.
    - Ante ``unknown``/``edge_only``, limpia en ``clear_after`` frames
      (no mantiene polígonos viejos).
    """

    def __init__(self, confirm: int = 2, clear_after: int = 2) -> None:
        self.confirm = confirm
        self.clear_after = clear_after
        self._hits = 0
        self._misses = 0
        self._last_good: BermEstimate | None = None

    def update(self, est: BermEstimate) -> BermEstimate:
        if est.status == "detected":
            self._hits += 1
            self._misses = 0
            self._last_good = est
            if self._hits >= self.confirm:
                return est
            # Aún no confirmado: no publicar polígono como hecho
            return BermEstimate(
                status="unknown",
                confidence=est.confidence,
                reason="esperando confirmación temporal de cordón",
                edge_polyline=list(est.edge_polyline),
            )

        self._hits = 0
        self._misses += 1
        if self._misses >= self.clear_after:
            self._last_good = None
        # No arrastrar detección previa
        return est
