# ── face-ml-worker ────────────────────────────────────────────────────────────
# Each RUN pip install is a separate cached layer.
# CPU-only torch is pre-installed so ultralytics doesn't pull 3GB of GPU libs.
FROM python:3.11-slim

WORKDIR /app

# OS dependencies needed by OpenCV, TensorFlow, and ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 1. OpenCV headless (~72 MB) — no display server needed in container
RUN pip install --no-cache-dir opencv-python-headless

# 2. TensorFlow (~620 MB) — install own layer so it's cached independently
RUN pip install --no-cache-dir tensorflow

# 3. tf-keras (small, depends on TF)
RUN pip install --no-cache-dir tf-keras

# 4. keras-facenet
RUN pip install --no-cache-dir keras-facenet

# 5. deepface
RUN pip install --no-cache-dir deepface

# 6. PyTorch CPU-only (~200 MB vs 3 GB for GPU version).
#    Install BEFORE ultralytics so pip sees torch already present
#    and skips downloading nvidia_cuda/cudnn/cublas packages.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# 7. ultralytics / YOLO — now uses the CPU torch already installed above
RUN pip install --no-cache-dir ultralytics

# 8. Remaining lightweight packages
RUN pip install --no-cache-dir fastapi uvicorn requests numpy

# Copy source + model weights
COPY src/ ./src/
COPY yolov8n-face.pt ./yolov8n-face.pt
COPY yolov8n.pt ./yolov8n.pt

EXPOSE 8001

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8001"]
