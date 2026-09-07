# Imagen orientada a evaluación Deliryum (Linux + NVIDIA CUDA).
# Build en Apple Silicon: docker build --platform linux/amd64 -t deliryum/bermguard:latest .

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLOT_CONFIG_DIR=/tmp/Ultralytics \
    ULTRALYTICS_OFFLINE=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pesos en BUILD TIME (evaluación sin red)
RUN mkdir -p /app/weights \
    && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); import pathlib, shutil; \
src=next(pathlib.Path('.').rglob('yolov8n.pt')); shutil.copy(src, '/app/weights/yolov8n.pt'); print('weights', src)"

ENV ULTRALYTICS_OFFLINE=1

COPY bermguard/ /app/bermguard/
COPY main.py /app/main.py
COPY README.md /app/README.md
COPY reporte_benchmark.md /app/reporte_benchmark.md

ENTRYPOINT ["python", "main.py"]
CMD ["--input", "/app/test", "--output", "/app/output", "--method", "1", "--weights", "/app/weights/yolov8n.pt"]
