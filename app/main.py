"""
Main FastAPI Application for purusharth-selfie-api.
Provides REST endpoints for R2 object exploration, indexing,
asynchronous batch jobs, selfie face matching, and test UI.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional
import uuid

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.face_engine import FaceEngineError, get_face_engine
from app.image_utils import (
    ImageValidationError,
    validate_and_decode_image,
    validate_object_key,
)
from app.index_store import index_store
from app.matcher import NoFaceDetectedError, face_matcher
from app.r2_client import (
    R2ClientError,
    R2ConfigurationError,
    R2ObjectNotFoundError,
    R2PayloadTooLargeError,
    r2_client,
)
from app.schemas import (
    AnalyzedFace,
    BatchIndexRequest,
    BatchJobStatus,
    BoundingBox,
    HealthResponse,
    IndexObjectRequest,
    IndexObjectResponse,
    JobErrorDetail,
    MatchResponse,
    R2ObjectItem,
    R2ObjectsResponse,
    SelfieAnalysisResponse,
)

# Configure structured logging without sensitive credentials
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("purusharth_api")

# In-memory batch job state tracker (in production can be backed by Redis / SQLite)
jobs_db: Dict[str, BatchJobStatus] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for initialization and graceful teardown."""
    logger.info("Initializing purusharth-selfie-api...")
    index_store.initialize()
    logger.info("Local SQLite index store initialized.")
    yield
    logger.info("Shutting down purusharth-selfie-api...")


app = FastAPI(
    title="purusharth-selfie-api",
    description="Face indexing and selfie matching pipeline for private Cloudflare R2 storage.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder for temporary web UI
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root():
    """Redirect or serve test UI."""
    return FileResponse(str(static_dir / "index.html"))


@app.api_route("/ui", methods=["GET", "HEAD"], include_in_schema=False)
async def test_ui():
    """Serve temporary test UI dashboard."""
    index_html = static_dir / "index.html"
    if not index_html.exists():
        raise HTTPException(status_code=404, detail="Test UI frontend file not found.")
    return FileResponse(str(index_html))


@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check system health, index store counts, and Cloudflare R2 connectivity.
    """
    r2_conn = r2_client.check_connection()
    engine = get_face_engine()
    stats = index_store.get_stats()

    return HealthResponse(
        status="ok",
        r2_configured=settings.is_r2_configured,
        r2_connection="connected" if r2_conn.get("accessible") else "disconnected",
        database_status="ready",
        model_adapter=engine.model_version,
        embedding_dimension=engine.embedding_dim,
        indexed_images_count=stats.get("indexed_images", 0),
        total_faces_count=stats.get("total_faces", 0),
        environment=settings.environment,
    )


@app.get("/r2/objects", response_model=R2ObjectsResponse, tags=["Cloudflare R2"])
async def list_r2_objects(
    prefix: str = Query(default="", description="Key prefix filter"),
    limit: int = Query(default=50, ge=1, le=500, description="Max objects to return"),
    cursor: Optional[str] = Query(default=None, description="Continuation token for pagination"),
):
    """
    List objects in the configured private Cloudflare R2 bucket with pagination.
    """
    if not settings.is_r2_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloudflare R2 credentials are not configured in environment.",
        )

    try:
        items, next_cursor, is_truncated = r2_client.list_objects_paginated(
            prefix=prefix,
            max_keys=limit,
            continuation_token=cursor,
        )

        # Cross-reference with index store to indicate indexed state
        object_items: List[R2ObjectItem] = []
        for it in items:
            record = index_store.get_indexed_image(it["key"])
            is_indexed = bool(record and record.get("status") == "indexed")
            object_items.append(
                R2ObjectItem(
                    key=it["key"],
                    size_bytes=it["size_bytes"],
                    last_modified=it["last_modified"],
                    etag=it["etag"],
                    is_indexed=is_indexed,
                )
            )

        return R2ObjectsResponse(
            objects=object_items,
            next_cursor=next_cursor,
            is_truncated=is_truncated,
            total_count=len(object_items),
        )
    except R2ClientError as err:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(err))


def process_single_r2_object(
    object_key: str,
    collection_scope: str = "default",
    force_reindex: bool = False,
) -> IndexObjectResponse:
    """Internal helper to index a single R2 image."""
    safe_key = validate_object_key(object_key)

    # 1. Inspect R2 object metadata
    metadata = r2_client.get_object_metadata(safe_key)
    etag = metadata.get("etag")
    size_bytes = metadata.get("size_bytes", 0)

    # 2. Check idempotency: skip if already indexed with identical ETag
    if not index_store.should_index(safe_key, etag, force_reindex=force_reindex):
        existing = index_store.get_indexed_image(safe_key)
        face_count = existing.get("face_count", 0) if existing else 0
        return IndexObjectResponse(
            object_key=safe_key,
            status="skipped",
            faces_detected=face_count,
            message="Object is already indexed and unchanged.",
        )

    # 3. Stream object safely from R2
    image_bytes = r2_client.download_object_bytes(safe_key)

    # 4. Validate and decode image
    pil_img, _ = validate_and_decode_image(image_bytes)

    # 5. Extract face embeddings
    engine = get_face_engine()
    faces = engine.detect_and_embed(pil_img)

    # 6. Save into index store
    status_str = "indexed" if len(faces) > 0 else "no_faces"
    index_store.record_indexed_image(
        image_key=safe_key,
        collection_scope=collection_scope,
        etag=etag,
        size_bytes=size_bytes,
        status=status_str,
        faces=faces,
        model_version=engine.model_version,
    )

    del image_bytes
    del pil_img
    import gc
    gc.collect()

    return IndexObjectResponse(
        object_key=safe_key,
        status=status_str,
        faces_detected=len(faces),
        message=f"Successfully processed image ({len(faces)} faces detected).",
    )


@app.post("/index/r2-object", response_model=IndexObjectResponse, tags=["Indexing"])
async def index_r2_object(payload: IndexObjectRequest):
    """
    Index a single image from Cloudflare R2 into the face database.
    """
    if not settings.is_r2_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloudflare R2 is not configured.",
        )

    try:
        return process_single_r2_object(
            object_key=payload.object_key,
            collection_scope=payload.collection_scope,
            force_reindex=payload.force_reindex,
        )
    except R2ObjectNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))
    except (ImageValidationError, ValueError) as err:
        index_store.record_failed_image(payload.object_key, payload.collection_scope, str(err))
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err))
    except R2PayloadTooLargeError as err:
        index_store.record_failed_image(payload.object_key, payload.collection_scope, str(err))
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(err))
    except Exception as err:
        logger.exception("Failed indexing object %s", payload.object_key)
        index_store.record_failed_image(payload.object_key, payload.collection_scope, str(err))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Indexing failed: {err}")


async def run_batch_indexing_job(job_id: str, payload: BatchIndexRequest):
    """Asynchronous worker for batch indexing with bounded concurrency."""
    job = jobs_db[job_id]
    job.status = "in_progress"
    job.updated_at = datetime.utcnow()

    try:
        # Collect target objects from R2
        all_objects = []
        continuation_token = None
        while True:
            items, continuation_token, is_truncated = r2_client.list_objects_paginated(
                prefix=payload.prefix,
                max_keys=100,
                continuation_token=continuation_token,
            )
            all_objects.extend(items)
            if payload.max_items and len(all_objects) >= payload.max_items:
                all_objects = all_objects[: payload.max_items]
                break
            if not is_truncated or not continuation_token:
                break

        job.total_items = len(all_objects)
        semaphore = asyncio.Semaphore(settings.max_batch_concurrency)

        async def worker(obj_meta):
            key = obj_meta["key"]
            async with semaphore:
                try:
                    # Run CPU/network intensive download & detection in executor
                    await asyncio.to_thread(
                        process_single_r2_object,
                        key,
                        payload.collection_scope,
                        payload.force_reindex,
                    )
                    job.successful_items += 1
                except Exception as ex:
                    logger.warning("Batch item %s failed: %s", key, ex)
                    job.failed_items += 1
                    job.errors.append(JobErrorDetail(object_key=key, error=str(ex)[:200]))
                finally:
                    job.processed_items += 1
                    job.updated_at = datetime.utcnow()

        await asyncio.gather(*(worker(item) for item in all_objects))
        job.status = "completed"
    except Exception as ex:
        logger.exception("Batch job %s aborted with error", job_id)
        job.status = "failed"
        job.errors.append(JobErrorDetail(object_key="*", error=str(ex)))
    finally:
        job.updated_at = datetime.utcnow()


@app.post("/index/r2-batch", response_model=BatchJobStatus, tags=["Indexing"])
async def trigger_batch_indexing(
    payload: BatchIndexRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger an asynchronous batch indexing job for R2 objects.
    Returns immediately with a job ID for progress tracking.
    """
    if not settings.is_r2_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloudflare R2 is not configured.",
        )

    job_id = str(uuid.uuid4())
    job = BatchJobStatus(
        job_id=job_id,
        status="pending",
        collection_scope=payload.collection_scope,
        total_items=0,
        processed_items=0,
        successful_items=0,
        failed_items=0,
        errors=[],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    jobs_db[job_id] = job

    background_tasks.add_task(run_batch_indexing_job, job_id, payload)
    return job


@app.get("/jobs/{job_id}", response_model=BatchJobStatus, tags=["Indexing"])
async def get_job_status(job_id: str):
    """
    Check the status and progress of an asynchronous batch indexing job.
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return jobs_db[job_id]


@app.post("/selfie/analyze", response_model=SelfieAnalysisResponse, tags=["Selfie"])
async def analyze_selfie(
    file: UploadFile = File(..., description="Selfie image to analyze"),
):
    """
    Analyze an uploaded selfie image:
    Detects faces, calculates bounding boxes, measures facial confidence,
    and validates if the image is suitable for recognition matching.
    """
    try:
        file_bytes = await file.read()
        pil_img, detected_format = validate_and_decode_image(file_bytes)
        width, height = pil_img.size

        engine = get_face_engine()
        raw_faces = engine.detect_and_embed(pil_img)

        # Sort faces by bounding box area to determine primary face
        def get_area(f):
            if f.bounding_box:
                return f.bounding_box.get("width", 0) * f.bounding_box.get("height", 0)
            return 0

        sorted_faces = sorted(raw_faces, key=get_area, reverse=True)
        analyzed_faces = []

        for idx, f in enumerate(sorted_faces):
            bbox = f.bounding_box or {"x": 0, "y": 0, "width": width, "height": height}
            analyzed_faces.append(
                AnalyzedFace(
                    face_index=f.face_index,
                    bounding_box=BoundingBox(**bbox),
                    confidence=round(f.confidence, 4),
                    is_primary=(idx == 0),
                    embedding_dim=len(f.embedding),
                )
            )

        face_count = len(analyzed_faces)
        is_valid = face_count == 1
        if face_count == 0:
            msg = "No face detected in the image. Please upload a clear photo with good lighting."
        elif face_count == 1:
            msg = "Single clear face detected. Suitable for matching."
        else:
            msg = f"{face_count} faces detected. Matching will compare candidate faces against the primary face."

        return SelfieAnalysisResponse(
            status="success" if face_count > 0 else "no_face_detected",
            image_width=width,
            image_height=height,
            format=detected_format,
            faces_detected_count=face_count,
            is_valid_selfie=is_valid,
            validation_message=msg,
            faces=analyzed_faces,
        )

    except ImageValidationError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        logger.exception("Error analyzing selfie image")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {err}")


@app.post("/selfie/match", response_model=MatchResponse, tags=["Matching"])
@app.post("/match/upload", response_model=MatchResponse, tags=["Matching"])
async def match_uploaded_selfie(
    file: UploadFile = File(..., description="Selfie photo file to match"),
    collection_scope: Optional[str] = Form(default=None, description="Event or collection identifier"),
    threshold: Optional[float] = Form(default=None, description="Optional cosine similarity threshold (default 0.45)"),
    top_k: int = Form(default=50, description="Max candidate photos to return"),
):
    """
    Upload a selfie image, analyze its facial embedding,
    and find matching candidate photographs in the private Cloudflare R2 bucket.
    """
    try:
        file_bytes = await file.read()
        pil_img, _ = validate_and_decode_image(file_bytes)

        response = face_matcher.match_selfie(
            selfie_image=pil_img,
            collection_scope=collection_scope,
            threshold=threshold,
            top_k=top_k,
        )
        return response

    except NoFaceDetectedError as err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err))
    except ImageValidationError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except Exception as err:
        logger.exception("Error in selfie match upload")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Matching failed.")


@app.post("/test/upload-photo", tags=["Testing"])
async def upload_test_photo(
    file: UploadFile = File(..., description="Image to upload directly to R2 bucket for testing"),
    key_prefix: str = Form(default="test-photos/", description="Folder prefix in R2"),
):
    """
    Convenience endpoint to upload a test photo directly to private R2 bucket
    for live verification and demonstration.
    """
    if not settings.is_r2_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloudflare R2 is not configured.",
        )

    try:
        file_bytes = await file.read()
        # Validate image format before storing
        _, detected_format = validate_and_decode_image(file_bytes)

        filename = file.filename or f"photo_{int(datetime.utcnow().timestamp())}.jpg"
        clean_key = f"{key_prefix.rstrip('/')}/{filename}"
        safe_key = validate_object_key(clean_key)

        mime = f"image/{detected_format.lower()}"
        res = r2_client.upload_object(safe_key, file_bytes, content_type=mime)
        return {
            "message": "Test photo uploaded to R2 successfully.",
            "key": res["key"],
            "etag": res["etag"],
            "size_bytes": res["size_bytes"],
        }
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))
