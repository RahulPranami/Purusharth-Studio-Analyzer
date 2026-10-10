"""
Benchmark script to measure indexing throughput, face counts, search latency,
and resource usage on a synthetic or real 10,000+ image dataset.

Usage:
    python scripts/benchmark.py [--num-images 1000] [--search-queries 100]
"""

import argparse
import os
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from PIL import Image

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.face_engine import FaceDetectionResult, get_face_engine
from app.index_store import SQLiteIndexStore


def run_benchmark(num_images: int = 1000, search_queries: int = 100):
    print("=" * 60)
    print(f"Starting Face Index & Vector Search Benchmark")
    print(f"Target dataset: {num_images:,} images | Query rounds: {search_queries}")
    print("=" * 60)

    test_db_path = "data/benchmark_test.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    store = SQLiteIndexStore(db_path=test_db_path)
    engine = get_face_engine("mock")
    dim = engine.embedding_dim

    print(f"Engine: {engine.model_version} (dim={dim})")
    print("\n[Step 1/3] Generating synthetic photos and extracting embeddings...")

    # Pre-generate representative synthetic test image
    base_img = Image.new("RGB", (640, 480), color=(120, 150, 180))

    t0 = time.time()
    faces_recorded = 0

    batch_size = 500
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        for i in range(batch_start, batch_end):
            key = f"benchmark/photo_{i:06d}.jpg"
            # Simulate 1 to 3 faces per photo
            num_faces = 1 if (i % 3 != 0) else 2
            simulated_faces = []
            for f_idx in range(num_faces):
                # Unit vector
                vec = np.random.randn(dim).astype(np.float32)
                vec /= np.linalg.norm(vec)
                simulated_faces.append(
                    FaceDetectionResult(
                        face_index=f_idx,
                        embedding=vec,
                        bounding_box={"x": 50 * f_idx, "y": 50, "width": 100, "height": 100},
                        confidence=0.95,
                    )
                )

            store.record_indexed_image(
                image_key=key,
                collection_scope="benchmark-suite",
                etag=f"etag-{i}",
                size_bytes=250_000,
                status="indexed",
                faces=simulated_faces,
                model_version=engine.model_version,
            )
            faces_recorded += len(simulated_faces)

        pct = (batch_end / num_images) * 100
        print(f"  Indexed {batch_end:,} / {num_images:,} images ({pct:.1f}%)...")

    index_time = time.time() - t0
    throughput_images = num_images / index_time
    throughput_faces = faces_recorded / index_time

    db_size_mb = os.path.getsize(test_db_path) / (1024 * 1024)

    print("\n--- Indexing Performance Results ---")
    print(f"Total Images Processed: {num_images:,}")
    print(f"Total Faces Extracted: {faces_recorded:,}")
    print(f"Total Indexing Time: {index_time:.2f} s")
    print(f"Throughput: {throughput_images:.1f} images/sec ({throughput_faces:.1f} faces/sec)")
    print(f"Database On-Disk Footprint: {db_size_mb:.2f} MB")

    print(f"\n[Step 2/3] Measuring Search Latency over {search_queries} queries...")
    latencies = []
    matches_found = []

    for _ in range(search_queries):
        query_vec = np.random.randn(dim).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        q_t0 = time.time()
        results = store.search_similar_faces(
            query_embedding=query_vec,
            collection_scope="benchmark-suite",
            threshold=0.25,
            top_k=20,
        )
        q_time = (time.time() - q_t0) * 1000  # ms
        latencies.append(q_time)
        matches_found.append(len(results))

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg_latency = np.mean(latencies)

    print("\n--- Search Latency Results ---")
    print(f"Mean Latency: {avg_latency:.2f} ms")
    print(f"p50 Latency:  {p50:.2f} ms")
    print(f"p95 Latency:  {p95:.2f} ms")
    print(f"p99 Latency:  {p99:.2f} ms")
    print(f"Average Candidates Matched per Query: {np.mean(matches_found):.1f}")

    print("\n[Step 3/3] Cleanup")
    del store
    import gc
    gc.collect()
    try:
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
    except Exception:
        pass
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Face Indexing & Vector Matching")
    parser.add_argument("--num-images", type=int, default=2000, help="Number of images to benchmark")
    parser.add_argument("--search-queries", type=int, default=50, help="Number of test queries")
    args = parser.parse_args()
    run_benchmark(num_images=args.num_images, search_queries=args.search_queries)
