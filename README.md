# BermGuard AI

Pipeline de visión computacional para **detección de maquinaria pesada (CAEX / bulldozer)**, **estimación de altura de pretil** respecto a la rasante operativa y **semáforo de proximidad** en botaderos mineros.

## Artefactos por video (`--output/<stem>_methodN/`)

1. `*_osd.mp4` — OSD: cajas, cresta/base del pretil, semáforo de proximidad  
2. `berm_height_vs_time.png` — altura de pretil vs tiempo  
3. `vehicle_spatial_distribution.png` — trayectoria espacial de vehículos  
4. `metadata.json` — FPS, tiempos, alertas, tasa de detección  

## Métodos (`--method`)

| Valor | Descripción |
|-------|-------------|
| `1` | YOLO vehículos + **YOLO-seg de pretil** (`weights/berm_yolov8n_seg.pt`; fallback clásico OpenCV si no hay pesos) |
| `2` | Solo OpenCV (CLAHE → gradientes → perfil cresta/rasante) |
| `all` | Ejecuta ambos |

## Métrica de altura

\[
H_{\mathrm{pretil}}(t) = y_{\mathrm{base}}(x,t) - y_{\mathrm{cresta}}(x,t)
\]

Referenciada al plano del suelo operativo (coords de imagen; `y` crece hacia abajo). Se resume por mediana a lo largo del cordón. **Metros** solo si se pasa `--meters-per-pixel` (calibración explícita); si no, se reporta en píxeles.

## Instalación local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python main.py --input data/samples --output output --method 1 --device cuda
# o: --device mps | cpu | auto
```

Flags relevantes:

- `--weights` — detector de vehículos (default `weights/yolov8n.pt`)
- `--berm-seg-weights` — segmentación de pretil (default `weights/berm_yolov8n_seg.pt`)
- `--meters-per-pixel` — escala monocular opcional
- `--max-frames` — depuración

## Docker (evaluación ciega)

Imagen Linux + NVIDIA CUDA. Pesos embebidos en **build** (sin descargas en runtime).

```bash
docker build -t bermguard:latest .

docker run --rm --gpus all \
  -v /ruta/local/test:/app/test \
  -v /ruta/local/output:/app/output \
  bermguard:latest \
  python main.py --input /app/test --output /app/output --method 1
```

Tag compatible con el brief: `deliryum/bermguard:latest` (alias opcional al build).

Apple Silicon (emulación amd64, CPU):

```bash
docker build --platform linux/amd64 -t bermguard:latest .
docker run --rm --platform linux/amd64 \
  -v "$PWD/data/samples:/app/test:ro" \
  -v "$PWD/output:/app/output" \
  bermguard:latest \
  python main.py --input /app/test --output /app/output --method 1 --device cpu
```

## Reentrenamiento de pretil (opcional)

```bash
python scripts/extract_frames.py --input data/samples --output data/berm_seg/raw_frames
python scripts/curate_labels.py
# o anotar a mano:
python scripts/annotate_berm.py --labels data/berm_seg/labels
python scripts/train_berm_seg.py --labels data/berm_seg/labels --device cuda --epochs 70
```

## Estructura

```
bermguard/
  main.py
  Dockerfile
  requirements.txt
  README.md
  reporte_benchmark.md
  bermguard/           # pipeline, detectores, pretil, viz
  scripts/             # extract / annotate / train seg
  weights/             # yolov8n.pt + berm_yolov8n_seg.pt
  output/              # artefactos de ejemplo (samples)
```

## Limitaciones (honestas)

- Altura monocular aproximada; no sustituye LiDAR/topografía.
- Detector vehicular COCO no fine-tuned a CAEX/bulldozer de faena.
- Segmentación de pretil entrenada sobre samples del brief; generaliza limitado a otras cámaras/ángulos.
- Sin calibración, el semáforo de proximidad opera en **píxeles**, no en metros absolutos.

Ver `reporte_benchmark.md` para comparación método 1 vs 2, FPS y recomendación de producción.
