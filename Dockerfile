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

# Limit thread pools to minimize RAM footprint on low-resource free containers
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Pre-download InsightFace buffalo_s model during build (<200MB RAM)
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection', 'recognition']); app.prepare(ctx_id=-1, det_size=(640,640))"

# Copy application source
COPY . .

# Create volume mount directory for SQLite index
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
