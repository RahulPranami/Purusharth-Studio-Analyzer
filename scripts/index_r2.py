"""
Standalone CLI script to index Cloudflare R2 bucket images into the face index.
Supports incremental indexing, prefix filtering, and bounded concurrency.

Usage:
    python scripts/index_r2.py [--prefix photos/] [--scope wedding] [--concurrency 5] [--force]
"""

import argparse
import asyncio
from datetime import datetime
import logging
import sys
import time
from typing import List

from app.config import settings
from app.face_engine import get_face_engine
from app.image_utils import validate_and_decode_image, validate_object_key
from app.index_store import index_store
from app.r2_client import r2_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("indexer_cli")


async def index_single_key(
    key: str,
    scope: str,
    force: bool,
    engine,
    semaphore: asyncio.Semaphore,
    stats: dict,
):
    async with semaphore:
        try:
            safe_key = validate_object_key(key)
            meta = await asyncio.to_thread(r2_client.get_object_metadata, safe_key)
            etag = meta.get("etag")
            size_bytes = meta.get("size_bytes", 0)

            # Check incremental indexing
            if not index_store.should_index(safe_key, etag, force_reindex=force):
                stats["skipped"] += 1
                return

            # Download chunked bytes
            data = await asyncio.to_thread(r2_client.download_object_bytes, safe_key)
            pil_img, _ = validate_and_decode_image(data)

            # Face extraction
            faces = engine.detect_and_embed(pil_img)
            status_str = "indexed" if faces else "no_faces"

            index_store.record_indexed_image(
                image_key=safe_key,
                collection_scope=scope,
                etag=etag,
                size_bytes=size_bytes,
                status=status_str,
                faces=faces,
                model_version=engine.model_version,
            )
            stats["successful"] += 1
            stats["faces_total"] += len(faces)
            logger.info("Indexed '%s': %d face(s)", safe_key, len(faces))

        except Exception as ex:
            stats["failed"] += 1
            logger.warning("Failed indexing '%s': %s", key, ex)
            index_store.record_failed_image(key, scope, str(ex)[:250])


async def main_async(args):
    if not settings.is_r2_configured:
        logger.error("R2 credentials not configured. Please check your .env file.")
        sys.exit(1)

    logger.info("Connecting to R2 bucket: %s", settings.r2_bucket_name)
    conn_check = r2_client.check_connection()
    if not conn_check.get("accessible"):
        logger.error("R2 bucket is not accessible: %s", conn_check.get("message"))
        sys.exit(1)

    engine = get_face_engine()
    logger.info("Using Face Engine: %s (%d-dim)", engine.model_version, engine.embedding_dim)

    # Fetch object keys
    all_keys: List[str] = []
    continuation_token = None
    logger.info("Listing objects with prefix '%s'...", args.prefix)

    while True:
        items, continuation_token, is_truncated = r2_client.list_objects_paginated(
            prefix=args.prefix,
            max_keys=100,
            continuation_token=continuation_token,
        )
        for it in items:
            # Filter image extensions
            k = it["key"].lower()
            if any(k.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                all_keys.append(it["key"])

        if args.limit and len(all_keys) >= args.limit:
            all_keys = all_keys[: args.limit]
            break
        if not is_truncated or not continuation_token:
            break

    logger.info("Found %d eligible images to process.", len(all_keys))
    if not all_keys:
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    stats = {"successful": 0, "failed": 0, "skipped": 0, "faces_total": 0}
    start_time = time.time()

    tasks = [
        index_single_key(
            key=k,
            scope=args.scope,
            force=args.force,
            engine=engine,
            semaphore=semaphore,
            stats=stats,
        )
        for k in all_keys
    ]

    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    logger.info("Indexing completed in %.2f seconds.", elapsed)
    logger.info(
        "Results: %d indexed, %d skipped, %d failed | Total faces extracted: %d",
        stats["successful"],
        stats["skipped"],
        stats["failed"],
        stats["faces_total"],
    )


def main():
    parser = argparse.ArgumentParser(description="Index Cloudflare R2 bucket images into face database.")
    parser.add_argument("--prefix", default="", help="R2 object key prefix filter")
    parser.add_argument("--scope", default="default", help="Collection scope identifier")
    parser.add_argument("--concurrency", type=int, default=5, help="Bounded concurrency level")
    parser.add_argument("--force", action="store_true", help="Force reindexing of already indexed images")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to index")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
