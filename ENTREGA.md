# Checklist de entrega — BermGuard AI

Conforme al brief Deliryum.AI (módulos + Docker plug & play).

## Contenido del paquete

- [x] Código modular (`bermguard/`, `main.py`, tipado)
- [x] `Dockerfile` con pesos embebidos (sin descargas en runtime; sin `ENTRYPOINT`)
- [x] `requirements.txt`
- [x] `README.md` (uso local + Docker + limitaciones + decisiones de ingeniería)
- [x] `reporte_benchmark.md` (método 1 vs 2, detector compartido, tabla por video)
- [x] `weights/yolov8n.pt` + `weights/berm_yolov8n_seg.pt`
- [x] `output/` de ejemplo sobre videos del brief (métodos 1 y 2): OSD + PNGs + `metadata.json`
- [x] Scripts de reentrenamiento de pretil (`scripts/`)

## Cómo evaluar (ciego)

```bash
docker build -t bermguard:latest .
# o: docker build -t deliryum/bermguard:latest .

docker run --rm --gpus all \
  -v /ruta/local/test:/app/test \
  -v /ruta/local/output:/app/output \
  bermguard:latest \
  python main.py --input /app/test --output /app/output --method 1
```

También: `--method 2` y `--method all`.

## Nota

Los videos de `data/samples` no se incluyen en el zip (peso / licencia). Montarlos en `--input` / volumen Docker.
