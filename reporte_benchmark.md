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

## Benchmark completo (Apple M4 · MPS · sin `--max-frames`)

Videos ~10 s c/u (`video_01` 302@30fps; `02–04` 240@24fps). Artefactos en `output/`.

| Video | Método | Throughput (FPS) | Altura media (m) | Alertas | Notas |
|-------|--------|------------------|------------------|---------|-------|
| video_01 | 1 | 14.38 | 0.99 | 297 | Banda crest/toe visible; muchas alertas PRETIL BAJO |
| video_01 | 2 | 22.49 | 0.95 | 266 | Solo geométrico |
| video_02 | 1 | 29.80 | 2.44 | 68 | Escala m/px por neumático (~0.081); serie ruidosa |
| video_02 | 2 | 51.18 | 1.09 | 156 | Fallback m/px fijo (~0.021) |
| video_03 | 1 | 31.72 | 1.16 | 136 | Polvo; conf. YOLO baja en algunos frames |
| video_03 | 2 | 50.77 | 0.93 | 208 | — |
| video_04 | 1 | 31.89 | 1.04 | 194 | Bulldozer a veces etiquetado como CAEX |
| video_04 | 2 | 49.88 | 1.20 | 129 | — |

Guía operativa registrada: `guideline_min_berm_m = 2.0` (50% diámetro neumático 4.0 m). Alerta PRETIL BAJO si estimación &lt; ~75% de esa guía.

### Validación Docker (Linux/amd64 + CUDA image)

- Build OK: `docker build --platform linux/amd64 -t deliryum/bermguard:latest .`
- Smoke OK (5 frames × 4 videos, `--device cpu`): OSD + plots + `metadata.json` en `output_docker/`
- Pin `numpy<2` requerido por PyTorch 2.2 de la imagen base
- En Mac no hay `--gpus`; el runner de evaluación Deliryum debe usar `docker run --gpus all`

VRAM CUDA: medir en el runner de evaluación con `nvidia-smi` (no disponible en Apple Silicon).

## Observaciones de calidad visual (OSD)

**Lo que funciona bien**

- Método 1 dibuja cajas CAEX, IDs de track, semáforo de proximidad y overlay `method`/`light`.
- En `video_01` la banda de pretil (crest/toe) se alinea con el cordón del botadero en varios frames (~0.8–1.1 m).
- Plots de trayectoria espacial (método 1) muestran paths coherentes en el tiempo.
- Throughput en régimen ≥ ~30 FPS (método 1) / ≥ ~50 FPS (método 2) en videos 02–04.

**Debilidades honestas**

- La altura monocular es **inestable frame-a-frame** (picos al techo ~3.5 m y caídas a pocos píxeles cuando el ridge se pierde por polvo o vehículos en primer plano).
- Escala `meters_per_pixel` depende del video (fallback fijo vs proxy de neumático) → medias no son estrictamente comparables entre clips.
- YOLO COCO confunde / pierde bulldozer bajo polvo; IDs de track se fragmentan → falsas alertas de proximidad crítica entre “dobles” del mismo vehículo.
- Método 2 no aporta valor de negocio en detección/proximidad; solo techo de FPS y ablación geométrica.

## Trade-offs

| Criterio | Método 1 | Método 2 |
|----------|----------|----------|
| Detección maquinaria | Sí | No |
| Semáforo proximidad | Sí | No (sin tracks) |
| FPS | Medio (~14–32 MPS) | Alto (~22–51 MPS) |
| VRAM | Requiere GPU en producción | CPU suficiente |
| Robustez noche/polvo | Mejor con CLAHE + detector | Frágil si el ridge desaparece |
| Altura pretil | Comparable + ancla a bbox | Baseline geométrico |

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

Mejoras prioritarias post-entrega (si hay iteración):

1. Suavizado temporal / Kalman en altura de pretil  
2. Fine-tune o clases mineras (CAEX vs bulldozer)  
3. Calibración de cámara o escala por escena fija para comparar videos  
4. TensorRT / half-precision en el runner CUDA  

En un despliegue industrial real, la medición metrológica del pretil debería respaldarse con **LiDAR / stereo**; la cámara aporta cobertura, tracking y alertas tempranas sobre infraestructura existente (alineado al producto Deliryum).

## Limitaciones honestas

- Escala monocular aproximada; no sustituye topografía.
- YOLO COCO no está fine-tuned a CAEX/bulldozer de faena.
- Videos de muestra sintéticos / controlados; la evaluación ciega puede diferir.
- Sin TensorRT en esta entrega (imagen CUDA runtime genérica).
