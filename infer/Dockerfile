FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

# Torch + torchvision CUDA 12.8 wheels
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

COPY infer_app /app/infer_app

ENV INFER_DATA_DIR=/app/data
RUN mkdir -p ${INFER_DATA_DIR}

EXPOSE 8001

CMD ["uvicorn", "infer_app.main:app", "--host", "0.0.0.0", "--port", "8001"]

