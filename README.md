# BermGuard AI

Pipeline de visión computacional para **detección de maquinaria pesada**, **estimación de altura de pretil** respecto a la rasante operativa y **semáforo de proximidad** en botaderos mineros.

## Arquitectura (comparación justa)

La detección y el tracking vehicular son **módulos compartidos**. Los métodos 1 y 2 solo cambian la estrategia de estimación del pretil:

| Valor | Pretil | Vehículos / proximidad / mapa espacial |
|-------|--------|----------------------------------------|
| `1` | YOLO-seg fine-tuned (`weights/berm_yolov8n_seg.pt`; fallback OpenCV si faltan pesos) | Compartido |
| `2` | OpenCV clásico (CLAHE → gradientes → perfil cresta/rasante) | Compartido |
| `all` | Ejecuta ambos | — |

Así el benchmark compara **únicamente** el estimador de pretil, no un pipeline incompleto vs uno completo.

## Artefactos por video (`--output/<stem>_methodN/`)

1. `*_osd.mp4` — OSD: cajas, cresta/base del pretil, semáforo de proximidad  
2. `berm_height_vs_time.png` — altura de pretil vs tiempo  
3. `vehicle_spatial_distribution.png` — trayectoria espacial (bottom-center de bbox)  
4. `metadata.json` — FPS end-to-end, tiempos, alertas, tasa de detección  

### Ejemplo de `metadata.json` (extracto)

```json
{
  "video": "video_01.mp4",
  "method": "1",
  "device": "mps",
  "resolution": {"width": 1920, "height": 1080},
  "frames_processed": 302,
  "wall_time_s": 59.263,
  "avg_throughput_fps": 5.096,
  "berm_detect_rate": 0.96,
  "mean_berm_height_px": 72.2,
  "scale_calibrated": false,
  "notes": "Detección/tracking vehicular compartidos entre métodos; ..."
}
```

## Métrica de altura

\[
H_{\mathrm{pretil}}(t) = y_{\mathrm{base}}(x,t) - y_{\mathrm{cresta}}(x,t)
\]

Referenciada al plano del suelo operativo (coords de imagen; `y` crece hacia abajo). Mediana a lo largo del cordón. **Metros** solo con `--meters-per-pixel` (escala local / escena rectificada). Una sola escala global **no** corrige perspectiva en toda la imagen.

## Detección vehicular (COCO)

YOLOv8n preentrenado en COCO detecta vehículos genéricos (`truck`, `bus`, `car`, …). Se normalizan como `heavy_vehicle`. **COCO no tiene clase bulldozer**: la diferenciación confiable CAEX vs bulldozer requiere fine-tuning minero (roadmap).

## Proximidad

Distancia entre **bottom-centers** de cada bbox (proxy de contacto con el suelo). Umbrales en píxeles = semáforo visual, no distancia métrica.

## Instalación local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python main.py --input data/samples --output output --method all --device auto
# o: --device mps | cuda | cpu
```

Flags:

- `--weights` — detector compartido (default `weights/yolov8n.pt`)
- `--berm-seg-weights` — seg de pretil (default `weights/berm_yolov8n_seg.pt`)
- `--meters-per-pixel` — conversión aproximada solo si hay escala local / ROI rectificada
- `--max-frames` — depuración

## Docker (evaluación ciega)

Imagen Linux + runtime NVIDIA CUDA. Pesos embebidos en **build** (sin descargas en runtime). **Sin `ENTRYPOINT`**: el comando del brief sobrescribe el `CMD` por defecto.

```bash
docker build -t bermguard:latest .

docker run --rm --gpus all \
  -v /ruta/local/test:/app/test \
  -v /ruta/local/output:/app/output \
  bermguard:latest \
  python main.py --input /app/test --output /app/output --method 1
```

Se validó build + ejecución funcional en CPU sobre `linux/amd64` (incluido video completo en smoke local). **No se dispuso de host NVIDIA** para validar CUDA end-to-end; la imagen usa runtime CUDA y depende del driver del host + NVIDIA Container Toolkit.

Apple Silicon (emulación amd64, CPU):

```bash
docker build --platform linux/amd64 -t bermguard:latest .
docker run --rm --platform linux/amd64 \
  -v "$PWD/data/samples:/app/test:ro" \
  -v "$PWD/output_docker:/app/output" \
  bermguard:latest \
  python main.py --input /app/test --output /app/output --method all --device cpu
```

## Reentrenamiento de pretil (opcional)

```bash
python scripts/extract_frames.py --input data/samples --output data/berm_seg/raw_frames
python scripts/curate_labels.py
python scripts/annotate_berm.py --labels data/berm_seg/labels
python scripts/train_berm_seg.py --labels data/berm_seg/labels --device cuda --epochs 70
```

## Decisiones de ingeniería

1. **Métricas relativas en px** sin calibración: evitar inventar metros monoculares.
2. **YOLO-seg de pretil**: el cordón real no es el horizonte; OpenCV queda como ablación/FPS.
3. **Detector compartido** entre métodos: el benchmark aísla el estimador de pretil.
4. **Bottom-center** para proximidad: mejor proxy de suelo que el centro del bbox.
5. **Producción real**: LiDAR/stereo para metrología; cámara para cobertura, tracking y alertas.

## Limitaciones

- Altura monocular aproximada; no sustituye topografía.
- Dataset de seg acotado a los 4 videos del brief → riesgo de sobreajuste.
- Split train/val por frames aleatorios (correlación temporal) → mAP no es generalización industrial.
- Sin calibración, proximidad y altura en **píxeles**.
- CUDA en Docker no validado en GPU NVIDIA física en esta entrega.

### Hardware

Benchmarks de entrega en Apple Silicon (MPS). Contenedor `linux/amd64` validado con smoke test CPU. CUDA configurada en la imagen, no validada end-to-end sin host NVIDIA.

Ver `reporte_benchmark.md` para la tabla completa por video, trade-offs y recomendación.
