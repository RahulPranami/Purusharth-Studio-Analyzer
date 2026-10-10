# Purusharth Studio: Face Recognition System Architecture & Integration Guide

## 1. Executive Summary

This document describes the end-to-end architecture, deployment configuration, and React Native / Expo integration of the **Purusharth Studio Face Recognition & Biometric Matching Pipeline**.

The system enables guests at weddings and luxury events to find their photos across multi-thousand photo galleries by capturing a single reference selfie. Photographs uploaded by studio admins are indexed into a vector database in real-time, and facial embeddings are matched using cosine similarity.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Storage & Cloud
        R2[("Cloudflare R2 Storage<br/>(purusharth-studio)")]
        Supabase[("Supabase PostgreSQL<br/>(events, media, profiles)")]
    end

    subgraph Backend ["FastAPI Analyzer Service (Render)"]
        API["FastAPI Endpoints<br/>(/selfie/analyze, /selfie/match, /index/r2-object)"]
        InsightFace["InsightFace Engine<br/>(SCRFD 500M + MobileFaceNet)"]
        VectorDB[("SQLite Vector Index Store<br/>(512-d unit embeddings)")]
        Matcher["Cosine Dot-Product Matcher<br/>(Threshold: 0.40)"]
    end

    subgraph Client ["Purusharth Studio App (Expo / React Native)"]
        AdminUpload["Admin Photo Uploads<br/>(src/app/admin/uploads.tsx)"]
        ProfileOnboard["Biometric Selfie Setup<br/>(src/app/profile.tsx)"]
        EventGallery["Event Photo Lookbook<br/>(src/app/event/[id].tsx)"]
        AnalyzerClient["Analyzer Service Client<br/>(src/services/analyzer.ts)"]
    end

    AdminUpload -->|1. Direct Upload Photo| R2
    AdminUpload -->|2. Register Metadata| Supabase
    AdminUpload -->|3. Trigger Indexing| AnalyzerClient
    AnalyzerClient -->|POST /index/r2-object| API
    API -->|Download & Detect Faces| R2
    API --> VectorDB

    ProfileOnboard -->|1. Validate Face Quality| AnalyzerClient
    AnalyzerClient -->|POST /selfie/analyze| API
    ProfileOnboard -->|2. Upload Selfie Reference| R2
    ProfileOnboard -->|3. Store selfie_key| Supabase

    EventGallery -->|1. Tap My Photos (AI)| AnalyzerClient
    AnalyzerClient -->|POST /selfie/match| API
    API -->|Fetch Event Embeddings & Match| VectorDB
    API --> Matcher
    Matcher -->|Return Matched Keys| AnalyzerClient
    AnalyzerClient -->|Filter Gallery Photos| EventGallery
```

---

## 3. Core Components

### 3.1. Face Analyzer Backend (`Purusharth-Studio-Analyzer`)
- **Technology Stack**: Python 3.12, FastAPI, ONNX Runtime (CPU), InsightFace (`buffalo_s`), SQLite.
- **Production URL**: `https://purusharth-studio-analyzer.onrender.com`
- **Hosting Environment**: Render Free Tier (512 MB RAM limit).
- **Face Model**:
  - Detection: SCRFD 500M (`det_500m.onnx`, 2.5 MB).
  - Recognition: MobileFaceNet (`w600k_mbf.onnx`, 13.6 MB).
  - Normalization: L2 unit normalization ($\|v\|_2 = 1$).
  - Similarity Metric: Cosine similarity via dot product ($u \cdot v \in [-1.0, 1.0]$).

### 3.2. Cloudflare R2 Object Storage
- **Account ID**: `c5b183bb011ad958f341adda9e499c13`
- **Bucket Name**: `purusharth-studio`
- **Public CDN Base**: `https://pub-9d4e540702114b88b33aac6b160ce50f.r2.dev`
- **Key Conventions**:
  - Event Photos: `events/{eventId}/{timestamp}_{filename}`
  - Client Selfies: `selfies/{userId}_{timestamp}_{filename}`

### 3.3. Client Mobile & Web App (`Purusharth-Studio`)
- **Technology Stack**: React Native, Expo SDK 52, Expo Router v4, Supabase JS, TypeScript.

---

## 4. Key Deployment Fixes & Optimizations

During deployment to Render's 512 MB RAM container environment, three critical bottlenecks were resolved:

1. **Model Pruning (137 MB $\rightarrow$ 13 MB)**:
   InsightFace's default initialization scans the model folder and instantiates all `.onnx` files before discarding unused modules. By pruning the 143 MB 3D landmark model (`1k3d68.onnx`) and attribute models during Docker build, peak ONNX initialization memory dropped by **90%** (from 137 MB down to 13.1 MB).
2. **Deterministic Root Storage (`INSIGHTFACE_ROOT`)**:
   Pinned model storage to `/app/models/insightface` with open read permissions (`chmod -R 777`) so non-root container users never attempt to re-download weights from Google Drive at runtime.
3. **Single-Threaded Execution**:
   Enforced single-thread environment variables (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) and capped batch concurrency to `2` to eliminate multi-threaded worker memory spikes.

---

## 5. API Reference

### 5.1. `GET /health`
Returns system diagnostics, model adapter, and index statistics.
```json
{
  "status": "ok",
  "r2_configured": true,
  "r2_connection": "connected",
  "database_status": "ready",
  "model_adapter": "insightface-buffalo_s-v1.0",
  "embedding_dimension": 512,
  "indexed_images_count": 1,
  "total_faces_count": 6,
  "environment": "production"
}
```

### 5.2. `POST /selfie/analyze`
Validates an uploaded selfie image for face detection and quality.
- **Request**: Multipart form-data with `file`.
- **Response**:
```json
{
  "faces_detected_count": 1,
  "is_valid_selfie": true,
  "faces": [
    {
      "face_index": 0,
      "bounding_box": { "x": 120, "y": 80, "width": 240, "height": 300 },
      "confidence": 0.985
    }
  ],
  "message": "Valid selfie image with 1 detected face."
}
```

### 5.3. `POST /selfie/match`
Matches an enrolled R2 selfie against indexed photos in a collection scope.
- **Request**:
```json
{
  "selfie_image_key": "selfies/usr_123_selfie.jpg",
  "collection_scope": "event_uuid_456",
  "threshold": 0.40,
  "top_k": 50
}
```
- **Response**:
```json
{
  "total_faces_in_query": 1,
  "matched_photos_count": 1,
  "threshold_applied": 0.4,
  "top_matches": [
    {
      "image_key": "events/event_uuid_456/17285623_photo.jpg",
      "best_similarity": 0.4696,
      "min_distance": 0.5304,
      "match_count": 1,
      "matched_faces": [
        {
          "image_key": "events/event_uuid_456/17285623_photo.jpg",
          "face_index": 4,
          "similarity": 0.4696,
          "distance": 0.5304,
          "bounding_box": { "x": 772, "y": 361, "width": 333, "height": 509 }
        }
      ]
    }
  ]
}
```

### 5.4. `POST /index/r2-object`
Indexes a single photo from Cloudflare R2 into the vector store.
- **Request**:
```json
{
  "object_key": "events/event_uuid_456/17285623_photo.jpg",
  "collection_scope": "event_uuid_456",
  "force_reindex": false
}
```
- **Response**:
```json
{
  "object_key": "events/event_uuid_456/17285623_photo.jpg",
  "status": "indexed",
  "faces_detected": 6,
  "message": "Successfully processed image (6 faces detected)."
}
```

---

## 6. React Native / Expo Code Integration

### 6.1. Environment Configuration
**File:** `Purusharth-Studio/.env.local`
```env
EXPO_PUBLIC_ANALYZER_API_URL=https://purusharth-studio-analyzer.onrender.com
```

### 6.2. Service Client Module
**File:** `Purusharth-Studio/src/services/analyzer.ts`
Provides four typed functions:
- `analyzeSelfie(imageInput)`
- `matchSelfieWithEvent({ selfieKey, collectionScope, threshold })`
- `indexPhotoInAnalyzer(objectKey, collectionScope)`
- `batchIndexEventPhotos(prefix, collectionScope)`

### 6.3. Profile Screen (Biometric Validation)
**File:** `Purusharth-Studio/src/app/profile.tsx`
- Runs `analyzeSelfie` before uploading.
- If zero faces detected: prompts user to retake with better lighting.
- If multiple faces detected: warns user to use a solo portrait.
- Saves `selfie_key` and `selfie_url` in Supabase `profiles`.

### 6.4. Event Detail Screen (Photo Lookbook)
**File:** `Purusharth-Studio/src/app/event/[id].tsx`
- Added **`✨ My Photos (AI)`** filter tab.
- Queries `matchSelfieWithEvent` using `profile.selfie_key`.
- Displays real-time matching status banner and match percentage badges on photos (`✨ 94% Match`).

### 6.5. Admin Upload Screen (Live Facial Indexing)
**File:** `Purusharth-Studio/src/app/admin/uploads.tsx`
- Replaced mock face counts with `indexPhotoInAnalyzer(uploadRes.key, selectedEventId)`.
- Updates upload queue item with true `faces_detected` count from the neural model.

---

## 7. Verification & Testing History

| Test Case | Environment | Result | Details |
|---|---|---|---|
| Friends Cast Photo (6 Faces) | Production Render | **PASS** | Detected all 6 faces accurately in 6 seconds |
| Ross Selfie Cosine Match | Production Render | **PASS** | Matched Ross with 0.4696 similarity (all other cast members $\le 0.12$) |
| Render RAM Utilization | Production Render | **PASS** | Maintained $< 160$ MB resident memory (limit: 512 MB) |
| React Native TypeScript Check | Expo Project | **PASS** | `npx tsc --noEmit` passed with 0 errors |
| Unit & Integration Tests | Local Pytest | **PASS** | 24 tests passed in 1.10s |
