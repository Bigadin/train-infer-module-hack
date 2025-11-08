# Base image with CUDA runtime for GPU support
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      git curl ca-certificates \
      libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install base Python deps first (excluding torch/torchvision here)
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install torch/torchvision CUDA wheels as requested
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Copy app
COPY app /app/app

ENV DATA_DIR=/app/data
RUN mkdir -p ${DATA_DIR}

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

