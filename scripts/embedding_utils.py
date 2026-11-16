from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def response_embeddings(response):
    values = getattr(response, "embeddings", None)
    if values is None and isinstance(response, dict):
        values = response.get("embeddings")
    if not values:
        raise RuntimeError("Ollama returned no embeddings")
    return values


def load_shared_embeddings(npz_path="data/processed/talk_embeddings.npz",
                           metadata_path="data/processed/talk_embeddings_metadata.csv"):
    bundle = np.load(npz_path, allow_pickle=False)
    vectors = bundle["vectors"].astype(np.float64)
    model_name = str(bundle["model_name"].item())
    talk_ids = bundle["talk_ids"].astype(str)
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix must be a finite 2D numeric array")
    if len(vectors) != len(metadata) or len(talk_ids) != len(metadata):
        raise ValueError("Embedding and metadata row counts do not match")
    if "talk_id" not in metadata:
        raise ValueError("Metadata is missing talk_id")
    if not np.array_equal(talk_ids, metadata["talk_id"].astype(str).to_numpy()):
        raise ValueError("Embedding IDs and metadata IDs are not in the same order")
    if "entry_type" in metadata and not metadata["entry_type"].eq("talk").all():
        raise ValueError("Shared embedding metadata should contain regular talks only")
    return vectors, metadata, model_name


def cosine_rank(query_vector, vectors, metadata, top_n=3):
    q = np.asarray(query_vector, dtype=float).reshape(1, -1)
    X = np.asarray(vectors, dtype=float)
    if X.ndim != 2 or q.shape[1] != X.shape[1]:
        raise ValueError("Query and corpus embeddings have different dimensions")
    if len(X) != len(metadata):
        raise ValueError("Metadata rows do not align with embedding rows")
    if not np.isfinite(X).all() or not np.isfinite(q).all():
        raise ValueError("Embeddings must contain only finite values")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    q = normalize(q)
    X = normalize(X)
    scores = (X @ q.T).ravel()
    result = metadata.copy()
    result["cosine_similarity"] = scores
    return result.sort_values("cosine_similarity", ascending=False).head(top_n).reset_index(drop=True)


def fit_reductions(vectors, random_state=386):
    import umap
    X = np.asarray(vectors, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError("PCA and UMAP require at least 3 rows and 2 embedding dimensions")
    pca = PCA(n_components=2, random_state=random_state)
    pca_xy = pca.fit_transform(X)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(10, X.shape[0] - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
        transform_seed=random_state,
    )
    umap_xy = reducer.fit_transform(X)
    return pca, pca_xy, reducer, umap_xy


def embed_question(question, model_name):
    import ollama
    question = str(question).strip()
    if not question:
        raise ValueError("Please enter a nonempty question")
    response = ollama.embed(
        model=model_name,
        input=f"search_query: {question}",
        truncate=False,
    )
    vectors = response_embeddings(response)
    if len(vectors) != 1:
        raise RuntimeError(f"Expected one query vector; received {len(vectors)}")
    query_vector = np.asarray(vectors[0], dtype=float)
    if query_vector.ndim != 1 or not np.isfinite(query_vector).all() or np.linalg.norm(query_vector) == 0:
        raise RuntimeError("Ollama returned an invalid query embedding")
    return normalize(query_vector.reshape(1, -1))[0]
