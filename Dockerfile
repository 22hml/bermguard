# Imagen orientada a evaluación Deliryum (Linux + NVIDIA CUDA).
# En Apple Silicon el smoke-test puede usar --platform linux/amd64 (emulación) o CPU.

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ULTRALYTICS_OFFLINE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Descargar pesos en BUILD TIME (cero descargas en runtime de evaluación)
RUN mkdir -p /app/weights \
    && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && mv /root/.config/Ultralytics 2>/dev/null || true \
    && find / -name 'yolov8n.pt' 2>/dev/null | head -1 | xargs -I{} cp {} /app/weights/yolov8n.pt \
    || python -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); import shutil; shutil.copy('yolov8n.pt','/app/weights/yolov8n.pt')"

COPY bermguard/ /app/bermguard/
COPY main.py /app/main.py
COPY README.md /app/README.md
COPY reporte_benchmark.md /app/reporte_benchmark.md

# Entrypoint flexible: el evaluador pasa --input/--output/--method
ENTRYPOINT ["python", "main.py"]
CMD ["--input", "/app/test", "--output", "/app/output", "--method", "1", "--weights", "/app/weights/yolov8n.pt"]
