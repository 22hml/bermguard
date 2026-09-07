# BermGuard AI

Pipeline de visión computacional para **detección de maquinaria pesada**, **estimación de altura de pretil** y **semáforo de proximidad** en botaderos mineros.

Prueba técnica — Deliryum.AI · Ingeniero/a de IA (Visión por Computadora).

## Qué hace

Para cada video en `--input` genera un subdirectorio en `--output` con:

1. Video OSD (`*_osd.mp4`) — cajas, máscara/líneas de pretil, semáforo de proximidad  
2. `berm_height_vs_time.png` — altura estimada del pretil vs tiempo  
3. `vehicle_spatial_distribution.png` — trayectoria espacial de vehículos  
4. `metadata.json` — FPS, tiempos, alertas  

## Métodos (`--method`)

| Valor | Descripción |
|-------|-------------|
| `1` | **YOLO (Ultralytics)** para vehículos + pretil clásico **anclado** a la base de bboxes |
| `2` | **Solo OpenCV** (CLAHE → Canny → morfología → perfil de cresta/rasante) |
| `all` | Ejecuta ambos |

## Modelo geométrico de altura (monocular)

En faena real, sistemas comerciales de *berm monitoring* suelen usar **LiDAR** (±10 cm). Aquí resolvemos una **aproximación monocular** usable con cámaras existentes:

1. Estimar **cresta** del pretil (borde superior en ROI inferior).  
2. Estimar **rasante** / suelo operativo (perfil + ancla a `y2` de vehículos en método 1).  
3. `height_px = ground_y - crest_y`  
4. `height_m ≈ height_px × meters_per_pixel`  

### Escala `meters_per_pixel`

Prioridad:

1. Flag `--meters-per-pixel` (calibración explícita)  
2. Proxy por tamaño de bbox CAEX/truck y diámetro típico de neumático (~4.0 m), asumiendo que el neumático ocupa ~30% de la altura del vehículo en vista 3/4  
3. Fallback: ~15 m de escena vertical / altura del frame  

### Umbral operativo de referencia

Buenas prácticas de botadero (cordón de seguridad): altura guía ≈ **50% del diámetro del neumático** del equipo que descarga. Se registra en `metadata.json` como `guideline_min_berm_m` y genera alerta si la estimación cae bajo ~75% de esa guía.

> **Limitación:** sin profundidad activa ni calibración intrínseca/extrínseca completa, el error absoluto en metros puede ser alto. Priorizamos **estabilidad temporal**, alertas relativas y trazabilidad del método.

## Requisitos locales (macOS / dev)

```bash
cd bermguard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Videos de muestra (no van en git):

```bash
mkdir -p data
ln -s "/ruta/a/Prueba Técnica - Pretil y Maquinaria Minera" data/samples
```

### Ejecutar

```bash
python main.py --input data/samples --output output --method 1 --device mps
# smoke rápido:
python main.py --input data/samples --output output --method 1 --max-frames 30 --device mps
```

## Docker (evaluación Deliryum)

Imagen pensada para **Linux + NVIDIA CUDA**. Los pesos se embeben en **build time** (sin descargas en runtime).

```bash
docker build -t deliryum/bermguard:latest .

docker run --rm --gpus all \
  -v /ruta/local/test:/app/test \
  -v /ruta/local/output:/app/output \
  deliryum/bermguard:latest \
  --input /app/test --output /app/output --method 1 --weights /app/weights/yolov8n.pt
```

En Apple Silicon el build CUDA puede requerir `--platform linux/amd64` (emulación, lento). Para smoke local preferir el venv nativo con MPS/CPU.

## Estructura

```
bermguard/
  main.py
  bermguard/
    pipeline.py
    preprocess.py
    detectors/yolo.py
    berm/{classical,yolo_berm}.py
    geometry/height.py
    tracking/proximity.py
    viz/{osd,plots}.py
  weights/          # yolov8n.pt en imagen Docker / local tras primer run
  Dockerfile
  reporte_benchmark.md
```

## Entrega

- Código tipado y modular  
- `README.md` (este archivo)  
- `Dockerfile`  
- `reporte_benchmark.md`  
- Carpeta `output/` con artefactos (generar antes de empaquetar; no versionar en git)

Repo privado: https://github.com/22hml/bermguard
