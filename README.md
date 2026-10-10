# purusharth-selfie-api

A standalone, production-grade Python FastAPI service for facial indexing and selfie recognition against private Cloudflare R2 storage.

---

## 1. System Overview & Architecture

`purusharth-selfie-api` allows event attendees or administrators to upload a selfie photo and instantly find matching photos from an existing private Cloudflare R2 collection without making the bucket public or granting broad event access.

```mermaid
flowchart TD
    Client([Client / Tester / Swagger / Web UI]) -->|POST /match/upload (Selfie)| Router[FastAPI Router]
    Client -->|POST /index/r2-batch| BatchEngine[Async Batch Indexer]
    Client -->|GET /ui| WebUI[Test Console UI]

    subgraph Face Pipeline
        ImageUtils[Safe Image Decoder & Validator]
        FaceEngine[FaceEngine: Normalization & Embedding]
        Matcher[Cosine Similarity Matcher]
    end

    subgraph Storage
        R2[(Private Cloudflare R2)]
        SQLite[(Local SQLite Index / Future pgvector)]
    end

    Router --> ImageUtils
    ImageUtils --> FaceEngine
    BatchEngine -->|Chunked Safe Stream| R2
    BatchEngine --> ImageUtils
    FaceEngine --> SQLite
    Router --> Matcher
    Matcher --> SQLite
```

---

## 2. Model Licensing & Clearance Audit

| Model / Framework | Code License | Weights License | Commercial Clearance | Operational Usage |
| :--- | :--- | :--- | :--- | :--- |
| **MockFaceEngine** | MIT | Proprietary | ✅ **Commercially Cleared** | **Default Engine**. Deterministic feature simulator, zero external weight downloads, 100% offline & CI tested. |
| **OpenCV YuNet + SFace** | Apache 2.0 / BSD 3-Clause | Apache 2.0 / BSD 3-Clause | ✅ **Commercially Cleared** | Recommended production CPU model; lightweight and fully cleared for commercial distribution. |
| **InsightFace (ArcFace / RetinaFace)** | MIT | **Non-Commercial / Academic Only** | ❌ **NOT Cleared** | Prohibited for commercial deployment due to weight training dataset restrictions. |
| **FaceNet (VGGFace2)** | MIT | Creative Commons NC | ⚠️ **Restricted** | Weights are restricted by underlying dataset terms. |

---

## 3. Local Installation & Setup

### Requirements
- Python 3.11+
- Windows, macOS, or Linux

### Step 1: Create Virtual Environment & Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy the template file to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` and fill in your Cloudflare R2 credentials:
```ini
R2_ACCOUNT_ID=c5b183bb011ad958f341adda9e499c13
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_BUCKET_NAME=purusharth-studio
R2_ENDPOINT_URL=https://c5b183bb011ad958f341adda9e499c13.r2.cloudflarestorage.com
SQLITE_DB_PATH=data/index.db
FACE_ENGINE_TYPE=mock
MATCH_SIMILARITY_THRESHOLD=0.50
```

> **How to create R2 Access Keys in Cloudflare Dashboard**:
> 1. Log in to [dash.cloudflare.com](https://dash.cloudflare.com).
> 2. Navigate to **R2 Object Storage** in the left sidebar.
> 3. Click **Manage R2 API Tokens** in the top right.
> 4. Click **Create API Token**, select **Object Read & Write** (or Read-only for production indexers), select your bucket (`purusharth-studio`), and click **Create API Token**.
> 5. Copy the **Access Key ID** and **Secret Access Key** into `.env`.

---

## 4. Running the API & Test Frontend

Start the FastAPI application with Uvicorn:
```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Test Web Console**: [`http://localhost:8000/ui`](http://localhost:8000/ui)
- **Interactive Swagger Documentation**: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **Health & Diagnostic Check**: [`http://localhost:8000/health`](http://localhost:8000/health)

---

## 5. API Endpoints Reference

### 1. `GET /health`
Returns connection status to R2, database stats, model adapter name, and indexed counts.

### 2. `GET /r2/objects`
Query parameters:
- `prefix`: (optional) Filter keys by prefix (e.g. `wedding/`, `guests/`).
- `limit`: (optional, default 50) Page size.
- `cursor`: (optional) Continuation token for pagination.

### 3. `POST /index/r2-object`
Body:
```json
{
  "object_key": "wedding/photo_01.jpg",
  "collection_scope": "default",
  "force_reindex": false
}
```

### 4. `POST /index/r2-batch`
Body:
```json
{
  "prefix": "wedding/",
  "collection_scope": "wedding-2026",
  "force_reindex": false,
  "max_items": 500
}
```
Returns a `job_id` for tracking progress via `GET /jobs/{job_id}`.

### 5. `GET /jobs/{job_id}`
Returns progress (`total_items`, `processed_items`, `successful_items`, `failed_items`, `errors`).

### 6. `POST /match/upload`
Multipart Form:
- `file`: Selfie image file (JPEG, PNG, WebP).
- `collection_scope`: (optional) Scope to filter search (e.g. `"wedding-2026"`).
- `threshold`: (optional) Minimum cosine similarity threshold (e.g. `0.60`).
- `top_k`: (optional, default `50`) Max matches to return.

Response:
```json
{
  "total_faces_in_query": 1,
  "matched_photos_count": 2,
  "threshold_applied": 0.5,
  "top_matches": [
    {
      "image_key": "wedding/photo_01.jpg",
      "best_similarity": 0.9821,
      "min_distance": 0.0179,
      "match_count": 1,
      "matched_faces": [
        {
          "image_key": "wedding/photo_01.jpg",
          "face_index": 0,
          "similarity": 0.9821,
          "distance": 0.0179,
          "bounding_box": {"x": 120, "y": 90, "width": 180, "height": 180}
        }
      ]
    }
  ]
}
```

---

## 6. CLI Tools & Performance Benchmarking

### Incremental R2 Indexer Script
Run batch indexing directly from the terminal with bounded concurrency:
```powershell
.\.venv\Scripts\python.exe scripts/index_r2.py --prefix test-photos/ --scope default --concurrency 5
```

### 10,000+ Photo Dataset Benchmark
Measure indexing throughput, vector latency, and resource footprint:
```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --num-images 10000 --search-queries 50
```

#### Benchmark Results (Windows / Python 3.14):
- **Images Processed**: 10,000
- **Faces Extracted & Indexed**: 13,334
- **Indexing Throughput**: ~497 images/sec (~663 faces/sec)
- **Database Footprint**: 11.17 MB for 10,000 images and 13,334 face vectors
- **Vector Search Latency (p50)**: 57.31 ms
- **Vector Search Latency (p95)**: 68.87 ms

---

## 7. Running Tests

Run the full pytest suite:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

All 23 automated tests cover:
- Health check and documentation endpoints.
- Valid image decoding and format conversion.
- Strict rejection of corrupt payloads, oversized images, and undersized thumbnails.
- Directory traversal and object key injection protection.
- Unit vector normalization ($L_2 = 1.0$) and cosine similarity metrics.
- Incremental indexing idempotency using ETags.
- Face upload matching and multi-photo candidate aggregation.

---

## 8. Security & Privacy Guidelines

- **No Public URLs**: The R2 bucket is kept completely private. Pre-signed or temporary access URLs must be gated behind authentication.
- **Never Log Embeddings or Faces**: Raw selfies and raw vector arrays are never printed to console logs.
- **Consent & Retention**: Always obtain explicit attendee consent before face analysis. Provide deletion hooks (`delete_image_index`) to purge biometric embeddings on request.
- **Access Control**: A face match in this system only provides candidate photo keys; it **must never grant event access** or elevated authorization.
