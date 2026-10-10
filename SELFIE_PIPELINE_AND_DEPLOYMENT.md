# Selfie Processing Pipeline & Deployment Guide

---

## Part 1: How the Selfie Analysis & Matching Pipeline Works

```mermaid
flowchart TD
    A[User Uploads Selfie Photo] --> B[Pillow Ingestion & EXIF Auto-Rotation]
    B --> C[SCRFD Deep CNN Multi-Scale Face Detector]
    C --> D[5-Point Facial Landmark Alignment: 112x112]
    D --> E[ArcFace Deep Residual Neural Network]
    E --> F[512-Dimensional Vector Extraction]
    F --> G[L2 Normalization: ||v|| = 1.0]
    G --> H[Vector Dot Product Cosine Similarity in SQLite]
    H --> I[Threshold Filtering: s >= 0.45]
    I --> J[Ranked Photo Matches Output]
```

### Step 1: Image Reception & Orientation Normalization
1. When a selfie is sent via `multipart/form-data`, the server verifies file size against safety limits (`15 MB`).
2. Pillow parses the binary header and inspects **EXIF Orientation Tags**. Smart-phones often save selfies rotated 90° or 270°; the pipeline applies `ImageOps.exif_transpose` to ensure faces are upright.
3. The image is validated for non-empty bytes and converted into a standard 3-channel RGB array.

### Step 2: Multi-Scale Face Detection (SCRFD)
1. The deep convolutional network **SCRFD** (Sample and Computation Redistribution for Efficient Face Detection) scans the image across multiple pyramid resolutions.
2. It detects faces even in challenging conditions: low light, tilted angles, non-frontal poses, or multiple faces in the background.
3. For each detected face, it computes:
   - A bounding box `[x, y, width, height]`
   - Detection confidence score (e.g. `0.76` – `0.98`)
   - **5 Key Facial Landmarks**: Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner.

### Step 3: 5-Point Alignment & Canonical Cropping
Human faces in selfies are rarely squared to the lens. The pipeline computes an **affine transformation matrix** from the 5 landmarks:
- Rotates the face so both eyes sit horizontally level.
- Centers the nose bridge and mouth.
- Crops and scales the face into a standardized **112 × 112 pixel** tensor. This guarantees that angle and distance variations do not disrupt recognition.

### Step 4: ArcFace Feature Embedding (512 Dimensions)
1. The aligned 112×112 face is passed through an **ArcFace** deep neural network (ResNet architecture).
2. The model maps invariant biometric features (eye distance, jawline curve, cheekbone height, nose structure) into points on a high-dimensional mathematical hypersphere.
3. Output: A 512-element list of 32-bit floating-point numbers:
   $$\mathbf{v} = [v_1, v_2, v_3, \dots, v_{512}]$$

### Step 5: Unit $L_2$ Normalization
The vector is normalized by its Euclidean length:
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{512} v_i^2}}$$
This ensures $\|\hat{\mathbf{v}}\|_2 = 1.0$.

### Step 6: Vector Cosine Similarity Search
When comparing the selfie vector $\mathbf{u}$ against an indexed photo's face vector $\mathbf{v}$:
Because both vectors have unit length ($\|\mathbf{u}\| = 1, \|\mathbf{v}\| = 1$), the cosine similarity is computed with a **single vector dot product**:
$$\text{Similarity } (s) = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{512} u_i \cdot v_i$$
$$\text{Distance } (d) = 1.0 - s$$

- **$s = 1.0$**: Exact duplicate image.
- **$s \ge 0.45$**: **Same person** across different photos, lighting, or angles. (In testing, Ross scored **`0.6271`**).
- **$s \le 0.15$**: Different individuals. (The other 5 Friends cast members scored **`-0.07` to `0.09`**).

### Step 7: Group Photo Aggregation & Ranking
If a photo in your Cloudflare R2 bucket contains 6 people:
1. The matcher calculates the dot product against all 6 indexed faces in that photo.
2. It picks the maximum similarity score (e.g., Ross at Face 3 with `0.6271`).
3. It associates that photo key (`photos/friends_cast.jpg`) with the highest score and ranks candidates in descending order.

---

## Part 2: API Endpoints Reference

### 1. `POST /selfie/analyze` (Inspect Face Before Searching)
- **Path**: `http://localhost:8000/selfie/analyze`
- **Method**: `POST`
- **Form Data**: `file: <selfie_file>`
- **Response**:
```json
{
  "status": "success",
  "image_width": 144,
  "image_height": 165,
  "format": "PNG",
  "faces_detected_count": 1,
  "is_valid_selfie": true,
  "validation_message": "Single clear face detected. Suitable for matching.",
  "faces": [
    {
      "face_index": 0,
      "bounding_box": { "x": 15, "y": 31, "width": 84, "height": 114 },
      "confidence": 0.7655,
      "is_primary": true,
      "embedding_dim": 512
    }
  ]
}
```

### 2. `POST /selfie/match` (Analyze & Search Photos)
- **Path**: `http://localhost:8000/selfie/match` *(or `/match/upload`)*
- **Method**: `POST`
- **Form Data**:
  - `file`: `<selfie_file>`
  - `threshold`: `0.45` (optional, default 0.45)
  - `collection_scope`: `"default"` (optional)
  - `top_k`: `50` (optional)
- **Response**:
```json
{
  "total_faces_in_query": 1,
  "matched_photos_count": 1,
  "threshold_applied": 0.45,
  "top_matches": [
    {
      "image_key": "photos/friends_cast.jpg",
      "best_similarity": 0.6271,
      "min_distance": 0.3729,
      "match_count": 1,
      "matched_faces": [
        {
          "image_key": "photos/friends_cast.jpg",
          "face_index": 3,
          "similarity": 0.6271,
          "distance": 0.3729,
          "bounding_box": { "x": 770, "y": 355, "width": 329, "height": 510 }
        }
      ]
    }
  ]
}
```

---

## Part 3: Where & How to Host This Application

Because this service runs **FastAPI + ONNX Runtime (CPU Machine Learning)**, it requires:
- **RAM**: Minimum 1 GB (2 GB recommended).
- **Disk**: ~1 GB for Python dependencies + downloaded ONNX model weights (`~300 MB`).
- **Network**: Outbound HTTPS access to Cloudflare R2.

Here are the 4 best hosting options:

---

### Option A: Railway.app (Recommended: Easiest 1-Click Cloud Deployment)
**Best for**: Instant setup, automatic GitHub deployments, automatic HTTPS, zero server management.
- **Cost**: Free trial / ~$5 per month.
- **Why it fits**: Railway handles Docker containers automatically, provides 8GB RAM, and doesn't time out during ML inference.

#### Step-by-Step:
1. Push this project to a GitHub repository:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit of purusharth-selfie-api"
   git remote add origin https://github.com/<your-username>/purusharth-selfie-api.git
   git push -u origin main
   ```
2. Log in to [Railway.app](https://railway.app).
3. Click **New Project** > **Deploy from GitHub repo** > select `purusharth-selfie-api`.
4. In Railway **Variables**, add:
   ```ini
   R2_ACCOUNT_ID=c5b183bb011ad958f341adda9e499c13
   R2_BUCKET_NAME=purusharth-studio
   R2_ACCESS_KEY_ID=your_access_key
   R2_SECRET_ACCESS_KEY=your_secret_key
   FACE_ENGINE_TYPE=insightface
   MATCH_SIMILARITY_THRESHOLD=0.45
   PORT=8000
   ```
5. Click **Generate Domain** (e.g. `purusharth-selfie-api.up.railway.app`).
6. Your API and Test UI are live at `https://purusharth-selfie-api.up.railway.app/ui`.

---

### Option B: Render.com (Simple Managed Web Service)
**Best for**: Reliable managed container hosting with free SSL.
- **Cost**: Free tier / $7/month starter tier.

#### Step-by-Step:
1. Go to [Render.com](https://render.com) and create an account.
2. Click **New +** > **Web Service**.
3. Connect your GitHub repository.
4. Settings:
   - **Environment**: `Python` (or `Docker`)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add your Environment Variables (`R2_ACCOUNT_ID`, `R2_BUCKET_NAME`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`).
6. Click **Deploy Web Service**.

---

### Option C: Hetzner / DigitalOcean VPS ($4–$6/month, Best Performance)
**Best for**: Maximum performance, zero cold starts, dedicated CPU cores, production scale.
- **Cost**: ~$4.50/month (Hetzner Cloud CX22: 2 vCPU, 4GB RAM) or $6/month (DigitalOcean Droplet).

#### Step-by-Step using Docker:
1. Create a `Dockerfile` (already provided below).
2. SSH into your VPS:
   ```bash
   ssh root@your-vps-ip
   ```
3. Clone your repository:
   ```bash
   git clone https://github.com/<your-username>/purusharth-selfie-api.git
   cd purusharth-selfie-api
   ```
4. Build and run the container with Docker:
   ```bash
   docker build -t purusharth-api .
   docker run -d --name selfie-api -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data purusharth-api
   ```
5. Put a reverse proxy like **Caddy** or **Nginx** in front for automatic HTTPS:
   ```text
   # Caddyfile example:
   api.purusharthstudio.com {
       reverse_proxy localhost:8000
   }
   ```

---

### Option D: Cloudflare Tunnel (Host from your own PC / Local Server for Free)
**Best for**: Testing remotely or developing without paying any hosting fees.
- **Cost**: **100% Free**.
- Uses Cloudflare's edge network to route internet traffic directly to your local FastAPI server running on `localhost:8000` through an encrypted tunnel.

#### Step-by-Step:
1. Run with `cf` CLI:
   ```powershell
   cf tunnel --url http://localhost:8000
   ```
2. Cloudflare will give you an instant public HTTPS URL (e.g. `https://random-words.trycloudflare.com`).
3. You can access the API and `/ui` console from anywhere in the world, securely connected to your private R2 bucket.

---

## Part 4: Production Dockerfile

Here is the production Dockerfile ready to be deployed on any cloud platform:

```dockerfile
FROM python:3.11-slim

# Install system libraries for OpenCV and ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace models during build so container boots instantly
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1, det_size=(640,640))"

# Copy application source
COPY . .

# Create volume mount directory for SQLite index
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
