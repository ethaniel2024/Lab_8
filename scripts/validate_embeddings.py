from __future__ import annotations

import json
from pathlib import Path

from embedding_utils import load_shared_embeddings

EXPECTED_MODEL = "nomic-embed-text:v1.5"
EXPECTED_TALKS = 33
EXPECTED_DIMENSIONS = 768


def main():
    vectors, metadata, model_name = load_shared_embeddings()
    if model_name != EXPECTED_MODEL:
        raise SystemExit(f"Expected {EXPECTED_MODEL}; found {model_name}")
    if vectors.shape != (EXPECTED_TALKS, EXPECTED_DIMENSIONS):
        raise SystemExit(
            f"Expected {(EXPECTED_TALKS, EXPECTED_DIMENSIONS)}; found {vectors.shape}"
        )
    if metadata["talk_id"].duplicated().any():
        raise SystemExit("Duplicate talk IDs found in metadata")
    run_path = Path("data/processed/embedding_run_metadata.json")
    if not run_path.exists():
        raise SystemExit(f"Missing run metadata: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("embedding_count") != EXPECTED_TALKS:
        raise SystemExit("Run metadata count does not match the embedding matrix")
    if run.get("embedding_dimensions") != EXPECTED_DIMENSIONS:
        raise SystemExit("Run metadata dimensions do not match the embedding matrix")
    print("Shared embedding cache passed validation.")
    print(f"Model: {model_name}")
    print(f"Shape: {vectors.shape}")
    print(f"Talks chunked: {int(metadata['was_chunked'].astype(bool).sum())}")


if __name__ == "__main__":
    main()
