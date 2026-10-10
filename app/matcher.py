"""
Matcher module responsible for comparing query face embeddings
against indexed faces and aggregating photo-level candidate rankings.
"""

from typing import Dict, List, Optional
from PIL import Image

from app.config import settings
from app.face_engine import FaceEngine, get_face_engine
from app.index_store import IndexStore, index_store
from app.schemas import AggregatedPhotoMatch, BoundingBox, FaceMatchItem, MatchResponse


class MatcherError(Exception):
    """Raised when matching fails."""
    pass


class NoFaceDetectedError(MatcherError):
    """Raised when no face is found in the uploaded selfie."""
    pass


class FaceMatcher:
    """Orchestrates selfie face extraction, vector search, and photo ranking."""

    def __init__(
        self,
        engine: Optional[FaceEngine] = None,
        store: Optional[IndexStore] = None,
    ):
        self._engine = engine
        self.store = store or index_store

    @property
    def engine(self) -> FaceEngine:
        return self._engine or get_face_engine()

    @engine.setter
    def engine(self, val):
        self._engine = val

    def match_selfie(
        self,
        selfie_image: Image.Image,
        collection_scope: Optional[str] = None,
        threshold: Optional[float] = None,
        top_k: int = 50,
    ) -> MatchResponse:
        """
        Extract face(s) from selfie, compare against stored embeddings,
        and rank matching photographs.
        """
        eff_threshold = threshold if threshold is not None else settings.match_similarity_threshold

        # Detect face(s) in selfie
        query_faces = self.engine.detect_and_embed(selfie_image)
        if not query_faces:
            raise NoFaceDetectedError("No face could be detected in the uploaded selfie.")

        # Aggregated photo matches keyed by image_key
        photo_map: Dict[str, Dict] = {}

        for query_face in query_faces:
            raw_matches = self.store.search_similar_faces(
                query_embedding=query_face.embedding,
                collection_scope=collection_scope,
                threshold=eff_threshold,
                top_k=top_k,
            )

            for match in raw_matches:
                img_key = match["image_key"]
                bbox_data = match.get("bounding_box")
                bbox = BoundingBox(**bbox_data) if bbox_data else None

                match_item = FaceMatchItem(
                    image_key=img_key,
                    face_index=match["face_index"],
                    similarity=match["similarity"],
                    distance=match["distance"],
                    bounding_box=bbox,
                )

                if img_key not in photo_map:
                    photo_map[img_key] = {
                        "image_key": img_key,
                        "best_similarity": match["similarity"],
                        "min_distance": match["distance"],
                        "matched_faces": [match_item],
                    }
                else:
                    curr = photo_map[img_key]
                    curr["matched_faces"].append(match_item)
                    if match["similarity"] > curr["best_similarity"]:
                        curr["best_similarity"] = match["similarity"]
                        curr["min_distance"] = match["distance"]

        # Convert to AggregatedPhotoMatch objects
        aggregated: List[AggregatedPhotoMatch] = []
        for p in photo_map.values():
            aggregated.append(
                AggregatedPhotoMatch(
                    image_key=p["image_key"],
                    best_similarity=p["best_similarity"],
                    min_distance=p["min_distance"],
                    match_count=len(p["matched_faces"]),
                    matched_faces=p["matched_faces"],
                )
            )

        # Sort aggregated photos descending by best similarity
        aggregated.sort(key=lambda x: x.best_similarity, reverse=True)

        return MatchResponse(
            total_faces_in_query=len(query_faces),
            matched_photos_count=len(aggregated),
            threshold_applied=eff_threshold,
            top_matches=aggregated[:top_k],
        )


# Global face matcher instance
face_matcher = FaceMatcher()
