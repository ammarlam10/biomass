# ─────────────────────────────────────────────────────────────────────────────
# Biomass regression pipeline
# Base: PyTorch 2.1.2 + CUDA 12.1 + cuDNN 8 (runtime image, no build tools)
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

# Minimal system dependencies (no GDAL needed; data is zarr not GeoTIFF)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── install Python dependencies (cached separately from code) ────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── copy project source (overridden by volume mount in docker-compose) ────────
COPY . .

# Default command – override in docker-compose or `docker run`
CMD ["python", "scripts/train.py", "--config", "configs/unet_resnet50.yaml"]
