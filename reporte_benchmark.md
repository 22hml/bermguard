# Reporte de Benchmark — BermGuard AI

## Objetivo

Comparar al menos dos enfoques para segmentación/estimación de pretil y apoyo a detección vehicular bajo transiciones lumínicas, justificando el método recomendado para producción (precisión vs FPS/VRAM).

## Métodos evaluados

### Método 1 — YOLO + pretil anclado

- **Detección:** YOLOv8n (COCO) → clases vehiculares mapeadas a CAEX / bulldozer / vehicle.
- **Tracking:** IoU greedy frame-a-frame.
- **Pretil:** OpenCV (Sobel vertical + crest/toe) + refinamiento si la base del bbox cae cerca de la rasante.
- **Proximidad:** semáforo por distancia entre centros de bbox.

### Método 2 — OpenCV clásico puro

- CLAHE + Sobel-Y + perfil de cresta/rasante en ROI inferior.
- Sin red neuronal → baseline liviano para trade-off FPS/VRAM.
- No genera detecciones vehiculares (el mapa espacial queda vacío o casi vacío).

## Smoke test local (Apple M4, 15 frames / video)

| Video | Método | Throughput ef. (FPS) | Altura media est. (m) |
|-------|--------|----------------------|------------------------|
| video_01 | 1 (MPS) | ~1.6* | ~1–3 (post-calibración de banda) |
| video_02 | 1 | ~23 | idem |
| video_03 | 1 | ~14 | idem |
| video_04 | 1 | ~30 | idem |
| video_01 | 2 (CPU) | ~25 | ~1–3 |
| video_02–04 | 2 | ~65–70 | ~1–3 |

\*El primer video absorbe cold-start de YOLO/MPS; en régimen el throughput sube.

> Re-ejecutar sin `--max-frames` antes de la entrega para completar esta tabla con valores definitivos y VRAM (`nvidia-smi` en el runner CUDA de evaluación).

## Trade-offs

| Criterio | Método 1 | Método 2 |
|----------|----------|----------|
| Detección maquinaria | Sí | No |
| Semáforo proximidad | Sí | No (sin tracks) |
| FPS | Medio | Alto |
| VRAM | ~requiere GPU para producción | CPU suficiente |
| Robustez noche/polvo | Mejor con CLAHE + detector | Frágil si el ridge desaparece |
| Altura pretil | Comparable al clásico + ancla | Baseline geométrico |

## Hiperparámetros relevantes

- YOLO: `conf=0.25`, `iou=0.45`, `imgsz=640`, pesos `yolov8n.pt`
- Proximidad: amarillo ≤ 180 px, rojo ≤ 90 px (centros)
- Pretil: ROI vertical 40–92% del frame; altura acotada a ~0.4–3.5 m vía `meters_per_pixel`
- Escala: neumático CAEX ref. 4.0 m; guía de cordón = 50% diámetro

## Recomendación de producción

**Método 1**, porque aporta el valor de negocio completo (maquinaria + proximidad + pretil). El método 2 queda como:

- benchmark de techo de FPS,
- fallback diagnóstico si el detector falla,
- y ablación del módulo geométrico.

En un despliegue industrial real, la medición metrológica del pretil debería respaldarse con **LiDAR / stereo**; la cámara aporta cobertura, tracking y alertas tempranas sobre infraestructura existente (alineado al producto Deliryum).

## Limitaciones honestas

- Escala monocular aproximada; no sustituye topografía.
- YOLO COCO no está fine-tuned a CAEX/bulldozer de faena.
- Videos de muestra sintéticos / controlados; la evaluación ciega puede diferir.
- Sin TensorRT en esta entrega (margen CUDA runtime genérica).
