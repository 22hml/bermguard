# Benchmark — BermGuard AI

## Objetivo

Comparar dos enfoques de estimación de pretil (y apoyo a detección vehicular) bajo variaciones de iluminación, y justificar el método recomendado para producción (precisión vs FPS/VRAM).

## Métodos

### Método 1 — YOLO vehículos + YOLO-seg pretil

- **Detección:** YOLOv8n (COCO) → clases vehiculares mapeadas a CAEX / bulldozer / vehicle.
- **Tracking:** IoU greedy frame-a-frame.
- **Pretil:** YOLOv8n-seg fine-tuned sobre frames anotados de los videos del brief (`weights/berm_yolov8n_seg.pt`). Cresta y base derivadas de la máscara;  
  \(H = y_{\mathrm{base}} - y_{\mathrm{cresta}}\) (mediana a lo largo del cordón).
- **Fallback:** si no hay pesos de seg, perfil clásico OpenCV (borde ≠ cordón; estados `detected` / `edge_only` / `unknown`).
- **Proximidad:** semáforo por distancia entre centros de bbox (umbrales en px sin calibración métrica).
- **Metros:** solo con `--meters-per-pixel` explícito.

### Método 2 — OpenCV clásico

- CLAHE + Sobel-Y + perfil de cresta/rasante en ROI.
- Sin red neuronal → baseline liviano (techo de FPS / ablación).
- No genera detecciones vehiculares (mapa espacial vacío o casi vacío).

## Resultados (samples del brief)

Corridas locales Apple Silicon (MPS) y validación Docker (linux/amd64, CPU smoke). Artefactos de referencia en `output/`.

### Throughput (método 1 vs 2, videos completos · MPS · corrida de entrega)

| Video | Método 1 (FPS) | Método 2 (FPS) | Detect rate M1 | Altura media M1 (px) |
|-------|----------------|----------------|----------------|----------------------|
| video_01 | ~5–14* | ~20 | ~96% | ~72 |
| video_02 | ~18 | ~34 | ~94% | ~44 |
| video_03 | ~19 | ~35 | ~78% | ~54 |
| video_04 | ~21 | ~36 | ~89% | ~55 |

\*Variación MPS/carga térmica en video_01 (1080p); en régimen suele situarse ~14 FPS. Método 2 es ~1.5–2× más rápido y no cubre maquinaria ni proximidad.

Validación seg (hold-out interno del dataset bootstrap): mask mAP50 del orden ~0.7–0.95 según partición; **N pequeño** → no interpretar como GT industrial.

### Docker

- Build: `docker build -t bermguard:latest .` (pesos en imagen; sin red en runtime).
- Smoke: `--device cpu --max-frames 5` en amd64.
- Pin `numpy>=1.24,<2` por compatibilidad con PyTorch 2.2 de la imagen base.

## Trade-offs

| Criterio | Método 1 | Método 2 |
|----------|----------|----------|
| Detección maquinaria | Sí | No |
| Semáforo proximidad | Sí | No |
| Pretil (seg entrenada) | Sí (si hay pesos) | Perfil clásico |
| FPS | Medio | Alto |
| VRAM | GPU recomendada | CPU suficiente |
| Robustez noche/polvo | Mejor | Frágil si el ridge desaparece |

## Hiperparámetros

- YOLO detect: `conf=0.25`, `iou=0.45`, `imgsz=640`, `yolov8n.pt`
- YOLO-seg pretil: `conf≈0.15`, filtro de espesor (rechaza muros/horizonte), enmascara vehículos
- Proximidad: amarillo ≤ 180 px, rojo ≤ 90 px (centros de bbox)
- Temporal: confirmación 2 frames / clear 2 frames
- Escala opcional: `--meters-per-pixel`; guía operativa 2.0 m (50% diámetro neumático 4.0 m) solo si hay metros

## Recomendación de producción

**Método 1**, porque cubre el flujo completo del brief (maquinaria + proximidad + pretil). El método 2 queda como:

- techo de FPS,
- fallback diagnóstico,
- ablación del módulo geométrico.

### Roadmap

1. Más anotaciones frame-a-frame y negativos (pista / polvo)  
2. Fine-tune detector a CAEX vs bulldozer  
3. Calibración de cámara o escala fija por escena  
4. TensorRT / half-precision en CUDA  

En faena, la metrología del pretil debería respaldarse con **LiDAR / stereo**; la cámara aporta cobertura, tracking y alertas sobre infraestructura existente.

## Limitaciones

- Altura monocular aproximada; no sustituye topografía.
- Dataset de seg acotado a los 4 videos del brief → riesgo de sobreajuste a esas cámaras.
- YOLO COCO no fine-tuned a equipos mineros; tracks pueden fragmentarse.
- Sin calibración, proximidad y altura en **píxeles**, no metros absolutos.
- Sin TensorRT en la imagen actual (CUDA runtime genérica).
