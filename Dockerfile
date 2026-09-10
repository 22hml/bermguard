# BermGuard — runtime Linux + NVIDIA CUDA.
# Apple Silicon: docker build --platform linux/amd64 -t bermguard:latest .

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    ULTRALYTICS_OFFLINE=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pesos en build time (imagen autocontenida, sin red en runtime)
COPY weights/yolov8n.pt /app/weights/yolov8n.pt
COPY weights/berm_yolov8n_seg.pt /app/weights/berm_yolov8n_seg.pt

ENV ULTRALYTICS_OFFLINE=1

COPY bermguard/ /app/bermguard/
COPY main.py /app/main.py
COPY README.md /app/README.md
COPY reporte_benchmark.md /app/reporte_benchmark.md

# Sin ENTRYPOINT: el brief puede hacer
#   docker run ... bermguard:latest python main.py --input ... --output ... --method 1
# Sin override se usa este CMD por defecto.
CMD ["python", "main.py", "--input", "/app/test", "--output", "/app/output", "--method", "1"]
