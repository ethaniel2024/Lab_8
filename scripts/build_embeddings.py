from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ollama
from sklearn.preprocessing import normalize

MODEL_DEFAULT = "nomic-embed-text:v1.5"
DOCUMENT_PREFIX = "search_document: "
# The Ollama registry build is configured for 2,048 tokens. A conservative
# character cap avoids relying on truncation or on model-specific tokenizers.
DEFAULT_CHUNK_CHARS = 4200
DEFAULT_OVERLAP_CHARS = 350


def response_embeddings(response) -> list[list[float]]:
    """Support current Ollama Pydantic responses and mapping-style responses."""
    values = getattr(response, "embeddings", None)
    if values is None and isinstance(response, dict):
        values = response.get("embeddings")
    if not values:
        raise RuntimeError("Ollama returned no embeddings. Check the model and Ollama service.")
    return values


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS,
               overlap_chars: int = DEFAULT_OVERLAP_CHARS) -> list[str]:
    """Split on sentence/word boundaries with a small overlap; cap is characters, not tokens."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    if max_chars <= overlap_chars:
        raise ValueError("max_chars must be greater than overlap_chars")
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            sentence_boundary = text.rfind(". ", start + int(max_chars * 0.60), end)
            if sentence_boundary > start:
                end = sentence_boundary + 1
            else:
                word_boundary = text.rfind(" ", start + int(max_chars * 0.60), end)
                if word_boundary > start:
                    end = word_boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def embed_one(text: str, model: str) -> np.ndarray:
    response = ollama.embed(
        model=model,
        input=f"{DOCUMENT_PREFIX}{text}",
        truncate=False,
    )
    vectors = response_embeddings(response)
    if len(vectors) != 1:
        raise RuntimeError(f"Expected one vector for one input; received {len(vectors)}")
    vector = np.asarray(vectors[0], dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise RuntimeError("Ollama returned an invalid embedding vector")
    return vector


def is_context_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(signal in message for signal in (
        "context", "too long", "maximum", "max tokens", "input length",
        "exceeds", "truncate", "token limit",
    ))


def embed_talk(text: str, model: str, max_chars: int, overlap_chars: int):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        raise ValueError("Encountered an empty talk after preprocessing")
    # First try the complete address. truncate=False prevents silent truncation.
    # Only a model/context-length error triggers chunking.
    try:
        vector = embed_one(text, model)
        vector = normalize(vector.reshape(1, -1))[0]
        return vector.astype(np.float32), 1
    except Exception as exc:
        if not is_context_error(exc):
            raise
    pieces = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
    vectors = [embed_one(piece, model) for piece in pieces]
    dimensions = {len(v) for v in vectors}
    if len(dimensions) != 1:
        raise RuntimeError(f"Embedding dimensions changed between chunks: {dimensions}")
    weights = np.asarray([max(1, len(piece.split())) for piece in pieces], dtype=np.float32)
    combined = np.average(np.vstack(vectors), axis=0, weights=weights)
    combined = normalize(combined.reshape(1, -1))[0]
    return combined.astype(np.float32), len(pieces)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def main():
    parser = argparse.ArgumentParser(description="Build shared Nomic Embed Text v1.5 vectors.")
    parser.add_argument("--input", default="data/processed/talks_october_2025_cleaned.csv")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    args = parser.parse_args()

    source = Path(args.input)
    data = pd.read_csv(source, keep_default_na=False)
    required = {args.text_column, "link", "talk_name", "speaker", "include_in_embeddings"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    data = data.loc[as_bool(data["include_in_embeddings"])].copy().reset_index(drop=True)
    data = data.loc[data[args.text_column].astype(str).str.strip().ne("")].copy().reset_index(drop=True)
    if data.empty:
        raise ValueError("No nonempty records remain after applying include_in_embeddings")
    if data["link"].duplicated().any():
        raise ValueError("Talk links must be unique because they are used as stable talk IDs")
    if not (data.get("entry_type", pd.Series("talk", index=data.index)).astype(str) == "talk").all():
        raise ValueError("Only entry_type='talk' records may be embedded")

    vectors, ids, chunk_counts = [], [], []
    for index, row in data.iterrows():
        vector, count = embed_talk(str(row[args.text_column]), args.model,
                                   args.chunk_chars, args.overlap_chars)
        vectors.append(vector)
        ids.append(str(row["link"]))
        chunk_counts.append(count)
        label = row["talk_name"]
        print(f"{index + 1}/{len(data)}: {label} ({count} chunk(s))")

    X = np.vstack(vectors).astype(np.float32)
    if args.model == MODEL_DEFAULT and X.shape[1] != 768:
        raise RuntimeError(
            f"{MODEL_DEFAULT} should return 768 dimensions, but Ollama returned {X.shape[1]}. "
            "Stop and verify the model tag before sharing the cache."
        )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "talk_embeddings.npz"
    np.savez_compressed(
        npz_path,
        vectors=X,
        talk_ids=np.asarray(ids, dtype=str),
        model_name=np.asarray(args.model),
    )

    metadata = data.drop(columns=[args.text_column, "text_raw"], errors="ignore").copy()
    metadata.insert(0, "talk_id", ids)
    metadata["embedding_chunks"] = chunk_counts
    metadata["was_chunked"] = [count > 1 for count in chunk_counts]
    metadata.to_csv(out / "talk_embeddings_metadata.csv", index=False, encoding="utf-8-sig")

    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    run_info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model,
        "embedding_dimensions": int(X.shape[1]),
        "embedding_count": int(X.shape[0]),
        "document_prefix": DOCUMENT_PREFIX.strip(),
        "query_prefix": "search_query:",
        "ollama_model_context_tokens": 2048 if args.model == MODEL_DEFAULT else None,
        "chunking": {
            "method": "sentence/word-boundary character chunks with overlap",
            "max_characters_per_chunk": args.chunk_chars,
            "overlap_characters": args.overlap_chars,
            "aggregation": "word-count-weighted mean followed by L2 normalization",
        },
        "source_file": str(source),
        "source_sha256": source_hash,
    }
    (out / "embedding_run_metadata.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {X.shape[0]} vectors with {X.shape[1]} dimensions to {npz_path}")
    print("Remember to commit the NPZ, aligned metadata CSV, and run metadata together.")


if __name__ == "__main__":
    main()
