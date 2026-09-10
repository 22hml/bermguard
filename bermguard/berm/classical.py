"""Segmentación / estimación de pretil — método clásico (OpenCV).

Diseño:
  1) Detectar *borde* de botadero (plataforma → talud). Eso NO es un pretil.
  2) Buscar *cordón* elevado cerca del borde (cresta y pie con evidencia propia).
  3) Si no hay evidencia suficiente → status=unknown (sin polígono inventado).
"""

from __future__ import annotations

import cv2
import numpy as np

from bermguard.types import BBox, BermEstimate


def _unknown(reason: str, *, edge: list[tuple[float, float]] | None = None) -> BermEstimate:
    return BermEstimate(
        status="edge_only" if edge else "unknown",
        confidence=0.0,
        reason=reason,
        edge_polyline=list(edge or []),
    )


def _prepare_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _vehicle_mask(h: int, w: int, detections: list[BBox] | None, pad: float = 0.04) -> np.ndarray:
    """Máscara True = píxel usable (fuera de vehículos)."""
    ok = np.ones((h, w), dtype=bool)
    if not detections:
        return ok
    for d in detections:
        px = pad * max(d.width, 1.0)
        py = pad * max(d.height, 1.0)
        x1, x2 = max(0, int(d.x1 - px)), min(w, int(d.x2 + px))
        y1, y2 = max(0, int(d.y1 - py)), min(h, int(d.y2 + py))
        ok[y1:y2, x1:x2] = False
    return ok


def _smooth_series(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if xs.size < 3:
        return ys.astype(np.float32)
    k = min(11, xs.size if xs.size % 2 == 1 else max(3, xs.size - 1))
    if k % 2 == 0:
        k -= 1
    pad = k // 2
    yp = np.pad(ys.astype(np.float32), (pad, pad), mode="edge")
    med = np.array([float(np.median(yp[i : i + k])) for i in range(ys.size)], dtype=np.float32)
    coef = np.polyfit(xs.astype(np.float64), med.astype(np.float64), 1)
    lin = np.polyval(coef, xs.astype(np.float64)).astype(np.float32)
    blend = 0.8 * lin + 0.2 * med
    lo, hi = float(np.percentile(med, 5)) - 6.0, float(np.percentile(med, 95)) + 6.0
    return np.clip(blend, lo, hi)


def _detect_dump_edge(
    gray: np.ndarray,
    usable: np.ndarray,
    *,
    y0: int,
    y1: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Borde plataforma→talud por columnas (pico de gradiente coherente)."""
    h, w = gray.shape
    roi = gray[y0:y1, :].astype(np.float32)
    rh = roi.shape[0]
    if rh < 16:
        return None

    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.abs(gy)
    mag = cv2.GaussianBlur(mag, (21, 3), 0)
    track_tex = np.abs(gx)
    track_tex = cv2.GaussianBlur(track_tex, (5, 5), 0)

    xs: list[float] = []
    ys: list[float] = []
    sc: list[float] = []
    for x in range(step, w - step, step):
        if not bool(usable[y0:y1, x].mean() > 0.55):
            continue
        col = mag[:, x]
        a, b = int(0.10 * rh), int(0.90 * rh)
        if b <= a + 4:
            continue
        local = col[a:b]
        j = int(np.argmax(local)) + a
        peak = float(col[j])
        med = float(np.median(local)) + 1e-6
        if peak < med * 2.8:
            continue
        up = float(roi[max(0, j - 10) : j, x].mean()) if j > 4 else float(roi[j, x])
        down = float(roi[j : min(rh, j + 18), x].mean())
        if up + 6.0 < down:
            continue
        # Penalizar huellas: textura horizontal fuerte en la "plataforma" sobre el borde
        tex = float(track_tex[max(0, j - 20) : j, max(0, x - 3) : min(roi.shape[1], x + 4)].mean())
        drop = up - down
        score = peak + 1.5 * drop - 0.85 * tex
        if score < med * 2.0:
            continue
        xs.append(float(x))
        ys.append(float(y0 + j))
        sc.append(score)

    if len(xs) < 12:
        return None

    xs_a = np.asarray(xs, dtype=np.float32)
    ys_a = np.asarray(ys, dtype=np.float32)
    sc_a = np.asarray(sc, dtype=np.float32)

    n = xs_a.size
    win = max(12, int(0.28 * n))
    best_i, best_s = 0, -1e9
    for i in range(0, n - win + 1):
        seg_y = ys_a[i : i + win]
        seg_s = sc_a[i : i + win]
        # Preferir tramos con variación espacial de y (bordes diagonales/reales)
        # y castigar y casi constante (huellas horizontales)
        y_span = float(seg_y.max() - seg_y.min())
        score = float(seg_s.mean()) + 0.15 * y_span - 0.5 * float(seg_y.std())
        if score > best_s:
            best_s, best_i = score, i
    sl = slice(best_i, best_i + win)
    xs_k, ys_k, sc_k = xs_a[sl], ys_a[sl], sc_a[sl]
    if float(sc_k.mean()) < float(np.percentile(sc_a, 40)):
        return None
    # Si el "borde" es casi una línea horizontal perfecta de huellas, rechazar
    if float(ys_k.max() - ys_k.min()) < 0.02 * h and float(np.std(ys_k)) < 0.008 * h:
        return None
    ys_s = _smooth_series(xs_k, ys_k)
    return xs_k, ys_s, sc_k


def _search_cordon(
    gray: np.ndarray,
    usable: np.ndarray,
    edge_xs: np.ndarray,
    edge_ys: np.ndarray,
    *,
    min_px: float,
    max_px: float,
) -> BermEstimate:
    """Busca cordón *arriba* del borde (lado plataforma) con cresta/pie independientes."""
    h, w = gray.shape
    score = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    score = cv2.GaussianBlur(score, (11, 5), 0)

    crest_xs: list[float] = []
    crest_ys: list[float] = []
    toe_ys: list[float] = []
    evidence: list[float] = []

    for x, ey in zip(edge_xs, edge_ys):
        xi = int(round(float(x)))
        ei = int(round(float(ey)))
        if xi < 2 or xi >= w - 2 or ei < 8 or ei >= h - 2:
            continue
        if not usable[max(0, ei - int(max_px)) : ei, xi].mean() > 0.4:
            continue

        # Buscar cresta EN LA PLATAFORMA (y < edge), no en el talud bajo el borde
        y_lo = max(2, ei - int(max_px))
        y_hi = max(y_lo + 4, ei - int(0.35 * min_px))
        if y_hi <= y_lo + 3:
            continue
        col = score[y_lo:y_hi, xi]
        if float(col.max()) < 1e-3:
            continue
        # Pico relativo
        j = int(np.argmax(col))
        if float(col[j]) < float(np.median(col)) * 1.8:
            continue
        cy = float(y_lo + j)
        # Pie: entre cresta y borde — mínimo de score / transición
        band = score[int(cy) : ei, xi]
        if band.size < 3:
            continue
        toe_off = int(np.argmin(band))
        # Preferir pie cerca del borde si el mínimo es ambiguo
        ty = float(cy + toe_off)
        if ei - ty < 0.25 * (ei - cy):
            ty = float(0.35 * cy + 0.65 * ei)
        height = ei - cy  # cordón respecto al borde
        if height < min_px or height > max_px:
            continue

        # Evidencia de material elevado: banda crest→toe más “estructurada” que pad sobre cresta
        above = gray[max(0, int(cy) - 12) : int(cy), max(0, xi - 2) : min(w, xi + 3)]
        body = gray[int(cy) : int(ty) + 1, max(0, xi - 2) : min(w, xi + 3)]
        if above.size < 4 or body.size < 4:
            continue
        # Cordón suele ser más oscuro/texturado que la plataforma lisa
        contrast = float(above.mean()) - float(body.mean())
        tex = float(body.std()) - 0.5 * float(above.std())
        # Penalizar si "arriba" también es pista muy texturada (huellas)
        gx = float(np.abs(cv2.Sobel(above.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)).mean()) if above.size > 8 else 0.0
        ev = contrast + 0.35 * tex - 0.04 * gx
        if ev < 6.0:  # umbral más estricto: sin montículo claro → no cordón
            continue

        crest_xs.append(float(xi))
        crest_ys.append(cy)
        toe_ys.append(ty)
        evidence.append(ev)

    if len(crest_xs) < 10:
        edge_poly = [(float(x), float(y)) for x, y in zip(edge_xs, edge_ys)]
        return _unknown(
            "borde detectado pero sin evidencia de cordón elevado",
            edge=edge_poly,
        )

    xs = np.asarray(crest_xs, dtype=np.float32)
    crest = _smooth_series(xs, np.asarray(crest_ys, dtype=np.float32))
    # Pie independiente suavizado (no offset fijo de cresta)
    toe = _smooth_series(xs, np.asarray(toe_ys, dtype=np.float32))
    toe = np.maximum(toe, crest + min_px * 0.5)
    # No empujar el pie por debajo del borde medio
    edge_interp = np.interp(xs, edge_xs, edge_ys)
    toe = np.minimum(toe, edge_interp + 2.0)

    heights = toe - crest
    height_px = float(np.median(heights))
    if height_px < min_px:
        return _unknown(
            "altura de cordón insuficiente tras filtrado",
            edge=[(float(x), float(y)) for x, y in zip(edge_xs, edge_ys)],
        )

    # Densificar
    x_d = np.linspace(float(xs[0]), float(xs[-1]), num=max(24, xs.size * 2))
    c_d = np.interp(x_d, xs, crest)
    t_d = np.interp(x_d, xs, toe)
    crest_poly = [(float(x), float(y)) for x, y in zip(x_d, c_d)]
    ground_poly = [(float(x), float(y)) for x, y in zip(x_d, t_d)]
    edge_poly = [(float(x), float(y)) for x, y in zip(edge_xs, edge_ys)]

    mask = np.zeros((h, w), dtype=np.uint8)
    cpts = np.array(crest_poly, dtype=np.int32)
    gpts = np.array(list(reversed(ground_poly)), dtype=np.int32)
    cv2.fillPoly(mask, [np.vstack([cpts, gpts])], 255)

    conf = float(min(0.92, np.mean(evidence) / 18.0))
    if conf < 0.45:
        return _unknown(
            f"confianza de cordón baja ({conf:.2f})",
            edge=edge_poly,
        )

    return BermEstimate(
        status="detected",
        crest_y=float(np.median(c_d)),
        ground_y=float(np.median(t_d)),
        height_px=height_px,
        height_m=None,  # metros solo si hay calibración explícita (pipeline)
        meters_per_pixel=None,
        scale_valid=False,
        confidence=conf,
        reason="cordón verificado sobre borde de botadero",
        mask=mask,
        crest_polyline=crest_poly,
        ground_polyline=ground_poly,
        edge_polyline=edge_poly,
    )


def estimate_berm_classical(
    frame_bgr: np.ndarray,
    meters_per_pixel: float | None = None,
    *,
    detections: list[BBox] | None = None,
    scale_valid: bool = False,
) -> BermEstimate:
    """Estima pretil con separación borde vs cordón.

    No inventa geometría: sin evidencia → ``unknown`` / ``edge_only``.
    ``height_m`` solo se rellena si ``scale_valid`` y hay ``meters_per_pixel``.
    """
    h, w = frame_bgr.shape[:2]
    # Bandas en píxeles (geométricas), no derivadas de un techo 0.14*h artificial de metros
    min_px = max(6.0, 0.012 * h)
    max_px = max(min_px + 4.0, 0.22 * h)

    gray = _prepare_gray(frame_bgr)
    usable = _vehicle_mask(h, w, detections)
    y0, y1 = int(0.12 * h), int(0.92 * h)
    step = max(3, w // 260)

    edge = _detect_dump_edge(gray, usable, y0=y0, y1=y1, step=step)
    if edge is None:
        return _unknown("sin borde de botadero coherente")

    edge_xs, edge_ys, _ = edge
    est = _search_cordon(
        gray,
        usable,
        edge_xs,
        edge_ys,
        min_px=min_px,
        max_px=max_px,
    )

    if est.status == "detected" and scale_valid and meters_per_pixel and meters_per_pixel > 0:
        est.meters_per_pixel = float(meters_per_pixel)
        est.scale_valid = True
        if est.height_px is not None:
            est.height_m = float(est.height_px * meters_per_pixel)
    elif est.status == "detected":
        est.scale_valid = False
        est.height_m = None
        est.meters_per_pixel = None
        est.reason = (est.reason + " | sin calibración métrica").strip(" |")

    return est
