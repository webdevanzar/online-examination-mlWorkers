# ── voice-ml-worker ───────────────────────────────────────────────────────────
# NOTE: This worker uses `sounddevice` to read from the host microphone.
# On Windows with Docker Desktop, expose the mic by running the container
# with --device or via the host-network mode set in docker-compose.yml.
# The container uses host networking so it can reach the host mic via PulseAudio
# or the Windows audio stack (WSL2 / Windows-native Docker).
FROM python:3.11-slim

WORKDIR /app

# OS deps required by sounddevice / PortAudio
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libportaudio2 \
        libportaudiocpp0 \
        portaudio19-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CPU-only slim wheel – voice VAD does not need GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
# torch is already installed above; avoid re-downloading the GPU wheel
RUN grep -v "^torch$" requirements.txt > requirements_fixed.txt \
    && pip install --no-cache-dir -r requirements_fixed.txt

# Copy source
COPY src/ ./src/

EXPOSE 8002

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8002"]
