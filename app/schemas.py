"""Pydantic schemas for request, response, and domain models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health and diagnostic response."""
    status: str = "ok"
    r2_configured: bool
    r2_connection: str
    database_status: str
    model_adapter: str
    embedding_dimension: int
    indexed_images_count: int
    total_faces_count: int
    environment: str


class R2ObjectItem(BaseModel):
    """Metadata item representing an R2 object."""
    key: str
    size_bytes: int
    last_modified: datetime
    etag: str
    is_indexed: bool = False


class R2ObjectsResponse(BaseModel):
    """Paginated list of objects in R2."""
    objects: List[R2ObjectItem]
    next_cursor: Optional[str] = None
    is_truncated: bool = False
    total_count: int


class IndexObjectRequest(BaseModel):
    """Request to index a single R2 object."""
    object_key: str = Field(..., description="Key of the object in R2 to index")
    collection_scope: str = Field(default="default", description="Event or collection identifier")
    force_reindex: bool = Field(default=False, description="Whether to reprocess unchanged images")


class IndexObjectResponse(BaseModel):
    """Response after indexing an object."""
    object_key: str
    status: str
    faces_detected: int
    message: str


class BatchIndexRequest(BaseModel):
    """Request to index objects matching a prefix or entire bucket."""
    prefix: str = Field(default="", description="Optional prefix filter for R2 keys")
    collection_scope: str = Field(default="default", description="Event or collection identifier")
    force_reindex: bool = Field(default=False, description="Whether to reprocess unchanged images")
    max_items: Optional[int] = Field(default=None, description="Max number of items to process")


class JobErrorDetail(BaseModel):
    """Error detail for a failed item in a batch job."""
    object_key: str
    error: str


class BatchJobStatus(BaseModel):
    """Status of an asynchronous batch indexing job."""
    job_id: str
    status: str  # "pending", "in_progress", "completed", "failed"
    collection_scope: str
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    errors: List[JobErrorDetail] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class BoundingBox(BaseModel):
    """Bounding box coordinates for a face."""
    x: int
    y: int
    width: int
    height: int


class FaceMatchItem(BaseModel):
    """Individual face match record."""
    image_key: str
    face_index: int
    similarity: float
    distance: float
    bounding_box: Optional[BoundingBox] = None


class AggregatedPhotoMatch(BaseModel):
    """Aggregated match result per photograph."""
    image_key: str
    best_similarity: float
    min_distance: float
    match_count: int
    matched_faces: List[FaceMatchItem]


class MatchResponse(BaseModel):
    """Response returned from selfie matching."""
    total_faces_in_query: int
    matched_photos_count: int
    threshold_applied: float
    top_matches: List[AggregatedPhotoMatch]


class AnalyzedFace(BaseModel):
    """Details of a single face detected during selfie analysis."""
    face_index: int
    bounding_box: BoundingBox
    confidence: float
    is_primary: bool
    embedding_dim: int


class SelfieAnalysisResponse(BaseModel):
    """Diagnostic response from analyzing an uploaded selfie."""
    status: str
    image_width: int
    image_height: int
    format: str
    faces_detected_count: int
    is_valid_selfie: bool
    validation_message: str
    faces: List[AnalyzedFace]

