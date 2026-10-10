FROM python:3.12-slim

# Install system libraries for OpenCV and ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Limit thread pools and configure explicit model location
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    INSIGHTFACE_ROOT=/app/models/insightface

# Pre-download InsightFace buffalo_s model to /app/models/insightface during build
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_s', root='/app/models/insightface', allowed_modules=['detection', 'recognition']); app.prepare(ctx_id=-1, det_size=(640,640))"

# Prune unused 143MB landmark (1k3d68.onnx) and auxiliary models so ONNX doesn't load them into RAM
RUN rm -f /app/models/insightface/models/buffalo_s/1k3d68.onnx \
          /app/models/insightface/models/buffalo_s/2d106det.onnx \
          /app/models/insightface/models/buffalo_s/genderage.onnx

# Ensure permissions for any execution user
RUN chmod -R 777 /app/models

# Copy application source
COPY . .

# Create volume mount directory for SQLite index with write permissions
RUN mkdir -p /app/data && chmod -R 777 /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
