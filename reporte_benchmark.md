# Benchmark — BermGuard AI

## Objetivo

Comparar dos estrategias de **estimación de pretil** bajo variaciones de iluminación, con **detección/tracking vehicular compartidos**, y justificar el método recomendado (precisión vs FPS/VRAM).

## Diseño experimental

> La detección y el tracking vehicular son módulos compartidos. Los métodos 1 y 2 comparan específicamente dos estrategias alternativas de estimación del pretil.

| Componente | Método 1 | Método 2 |
|------------|----------|----------|
| YOLO vehículos (COCO → `heavy_vehicle`) | Sí | Sí |
| Tracking IoU + proximidad (bottom-center) | Sí | Sí |
| Mapa espacial / OSD / metadata | Sí | Sí |
| Pretil | YOLO-seg fine-tuned | OpenCV clásico |

### Método 1 — YOLO-seg pretil

- Pesos: `weights/berm_yolov8n_seg.pt` (YOLOv8n-seg sobre frames anotados del brief).
- Cresta/base derivadas de la máscara; \(H = y_{\mathrm{base}} - y_{\mathrm{cresta}}\) (mediana del cordón).
- Fallback clásico si faltan pesos.
- Filtro de espesor + restar máscaras de vehículos + filtro temporal (confirm 2 / clear 2).

### Método 2 — OpenCV clásico (ablación)

- CLAHE + Sobel-Y + perfil cresta/rasante en ROI.
- Mismo detector: aísla el efecto del estimador de pretil y aporta techo de FPS relativo.

### Detección vehicular (honestidad)

YOLOv8n-COCO no clasifica CAEX vs bulldozer. Las detecciones compatibles se etiquetan `heavy_vehicle`. Diferenciación minera fiable → fine-tune dedicado (roadmap).

### Proximidad y metros

- Proximidad: distancia entre bottom-centers (px). Proxy visual, no métrica.
- `--meters-per-pixel`: conversión aproximada solo con escala local / escena rectificada; **no** corrige perspectiva global.

## Resultados (corrida de entrega · MPS · detector compartido · `output/`)

Fuente: `metadata.json` por corrida. `avg_throughput_fps` = **FPS end-to-end** (decode + inferencia + OSD + encode MP4), no solo latencia del modelo.

| Video | Resolución | Frames | Método | FPS end-to-end | Detect rate pretil | Altura media (px) | wall_time_s |
|-------|------------|-------:|--------|---------------:|-------------------:|------------------:|------------:|
| video_01 | 1920×1080 | 302 | 1 | 5.096 | 96.0% | 72.2 | 59.263 |
| video_01 | 1920×1080 | 302 | 2 | 16.826 | 47.4% | 29.3 | 17.948 |
| video_02 | 1280×720 | 240 | 1 | 19.285 | 93.8% | 43.5 | 12.445 |
| video_02 | 1280×720 | 240 | 2 | 24.941 | 54.2% | 17.1 | 9.623 |
| video_03 | 1280×720 | 240 | 1 | 19.854 | 77.9% | 53.5 | 12.088 |
| video_03 | 1280×720 | 240 | 2 | 27.536 | 28.3% | 33.1 | 8.716 |
| video_04 | 1280×720 | 240 | 1 | 20.957 | 88.7% | 55.4 | 11.452 |
| video_04 | 1280×720 | 240 | 2 | 25.223 | 39.6% | 24.4 | 9.515 |

Ratio FPS M2/M1 (misma corrida, detector compartido): video_01 \(16.826/5.096 \approx 3.30\times\); 720p \(\approx 1.20\text{–}1.39\times\). El sobrecosto de M1 es principalmente la segmentación de pretil.

### Entrenamiento YOLO-seg (verificación funcional)

- Épocas / imgsz / batch / seed: ver `scripts/train_berm_seg.py` (defaults: 80, 640, 4, seed=42).
- Split: aleatorio por frames (`val_ratio=0.2`) sobre las mismas cuatro secuencias.
- **Limitación:** frames consecutivos están altamente correlacionados → riesgo de *data leakage* temporal. El mAP de validación **no** es una estimación confiable de generalización industrial; se reporta solo como chequeo de que el entrenamiento convergió.
- Preferible en iteraciones futuras: leave-one-video-out (entrenar 01–03, validar 04).

## Trade-offs

| Criterio | Método 1 (seg) | Método 2 (OpenCV) |
|----------|----------------|-------------------|
| Tasa de detección de pretil | Alta en samples | Menor / más frágil |
| Robustez noche/polvo | Mejor | Depende del ridge |
| FPS end-to-end | Menor (costo seg) | Mayor |
| VRAM | GPU recomendada | Misma detección + OpenCV liviano |

## Hiperparámetros

- YOLO detect: `conf=0.25`, `iou=0.45`, `imgsz=640`, `yolov8n.pt`
- YOLO-seg pretil: `conf≈0.15`, filtro de espesor, enmascara vehículos
- Proximidad: amarillo ≤ 180 px, rojo ≤ 90 px (bottom-centers)
- Temporal: confirmación 2 / clear 2 frames
- Guía operativa 2.0 m (50% de Ø neumático 4.0 m) **solo** si hay metros calibrados

## Docker

- Build: `docker build -t bermguard:latest .` (pesos en imagen; sin red en runtime).
- `CMD` por defecto; **sin `ENTRYPOINT`** → compatible con `docker run … python main.py …`.
- Validado: build + CPU en `linux/amd64`. **CUDA end-to-end no validado** en host NVIDIA físico.
- Pin `numpy>=1.24,<2` por compatibilidad con PyTorch 2.2 de la imagen base.

## Recomendación de producción

**Método 1** para el estimador de pretil (mejor detect rate y robustez en samples). Método 2 como ablación y referencia de costo/FPS del módulo geométrico.

### Roadmap

1. Más anotaciones y negativos (pista / polvo); split por video  
2. Fine-tune detector a CAEX vs bulldozer  
3. Calibración / homografía por escena  
4. TensorRT / half-precision en CUDA  
5. LiDAR / stereo para metrología absoluta  

## Limitaciones

- Altura monocular aproximada; no sustituye topografía.
- Dataset de seg acotado a los 4 videos del brief.
- COCO → `heavy_vehicle` sin taxonomía minera.
- Sin calibración: proximidad y altura en píxeles.
