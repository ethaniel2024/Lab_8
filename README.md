# Lab 8 — Text Embeddings, Semantic Search, and Dimensionality Reduction

## Overview

In this final lab, you will explore how text can be represented as numerical vectors. You will use a shared set of instructor-generated embeddings for 33 regular talks in the supplied October 2025 General Conference sample, fit PCA and UMAP, embed a gospel-related question, and retrieve the three talks most similar to that question.

**Important:** semantic similarity is not a measure of doctrinal correctness, spiritual authority, or proof that a talk directly answers a question. Treat retrieval results as leads for human reading and interpretation.

**Instructor release gate:** this ZIP contains the student materials and cleaned corpus, but not the real embedding cache. Before distributing it to students, run the instructor build steps below, run `uv run python scripts/validate_embeddings.py`, and include the three generated embedding artifacts in the student release.

## Corpus and preprocessing

The supplied student-produced Lab 6 file contains 35 records:

- 33 regular conference talks — included in the embedding corpus;
- 1 conference introduction — retained in the CSV but excluded from embeddings;
- 1 sustaining-of-authorities administrative item — retained in the CSV but excluded from embeddings.

The processed file keeps all 35 records and includes an `include_in_embeddings` flag. For each regular talk, the publisher-provided opening synopsis is stored separately in `summary`; `text` contains the spoken address body used for embeddings. The original scraped text is retained in `text_raw` for auditability. `word_count` is computed from the cleaned spoken body, while `source_word_count` preserves the source file's count.

The source CSV has no date values. Dates are left blank rather than inferred. The file is a student-produced sample, not a verified complete archive of every October 2025 address. An alternate 35-row student file was checked and had the same links and transcript text; a separate 794-row file covered 2010–2020 and was excluded because it is a different corpus. See `data/processed/dataset_cleaning_audit.md` for the source comparison, row-level inclusion, and remaining caveats.

## Model choice and verified configuration

This lab uses **Nomic Embed Text v1.5** through Ollama:

```text
nomic-embed-text:v1.5
```

The Ollama model registry lists this build with a 2,048-token context and 768-dimensional output. The upstream model card documents task prefixes: `search_document:` for corpus documents and `search_query:` for user queries. The build script uses those prefixes consistently. Because the Ollama package has a 2K context, long addresses are split conservatively into overlapping character chunks before embedding; they are not silently truncated. Chunk vectors are combined with a word-count-weighted mean and L2 normalization.

References:

- [Ollama model page and model metadata](https://ollama.com/library/nomic-embed-text:v1.5)
- [Nomic Embed Text v1.5 model card and task prefixes](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Ollama embedding API](https://docs.ollama.com/api/embed)

## Instructor: generate the shared embeddings

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/download), and [Quarto](https://quarto.org/docs/get-started/).

From the Lab 8 project directory:

```bash
ollama pull nomic-embed-text:v1.5
uv sync
uv run python scripts/build_embeddings.py
uv run python scripts/validate_embeddings.py
```

The script filters to `include_in_embeddings == True`, verifies that only regular talks are included, uses each talk's `link` as its stable ID, and saves three aligned files:

- `data/processed/talk_embeddings.npz` — the numeric embedding matrix, talk IDs, and model name;
- `data/processed/talk_embeddings_metadata.csv` — one metadata row per vector, in the same order;
- `data/processed/embedding_run_metadata.json` — model, dimensions, chunking settings, timestamp, and source-file hash.

Commit all three generated files together. Do not mix an NPZ file with a metadata CSV generated from a different run or dataset. The script reports the actual vector dimension returned by Ollama; for the specified model tag, expect 768.

The script first attempts each complete address with truncation disabled. Only if Ollama reports a context-length error does it split that address into chunks (default maximum 4,200 characters, 350-character overlap). This is a conservative character-based safeguard for the Ollama build's 2K token context, not a claim that characters equal tokens. Chunk vectors are combined into one vector for that talk. Adjust only if you deliberately change the model/build and regenerate the complete corpus.

**Release gate:** the shared embedding files are generated locally and are not bundled in this starter ZIP. Do not distribute the student copy until the instructor has run the build script successfully and committed the three generated files listed above.

## Student setup

Students do not need to regenerate the corpus embeddings, but they do need Ollama and the same model to embed their own question.

```bash
ollama pull nomic-embed-text:v1.5
uv sync
```

Install Quarto separately if you want to render the `.qmd` to HTML. Then run:

```bash
uv run quarto render lab_8.qmd
```

If Quarto does not automatically find the environment's Python kernel, register it once:

```bash
uv run python -m ipykernel install --user --name stat386-lab8
```

Do not substitute another embedding model for the query. The query vector must be generated by the same model and dimensions as the shared talk vectors.

## Student workflow

1. Load the shared embedding matrix and aligned metadata.
2. Inspect the number of talks and embedding dimensions.
3. Fit PCA and visualize the first two principal components.
4. Fit UMAP and visualize its two-dimensional layout.
5. Write your own gospel-related question and embed it with the same model using `search_query:`.
6. Rank talks by cosine similarity in the **original embedding space** and report the top three.
7. Transform the query into the already-fitted PCA and UMAP spaces and plot it as a separate marker.
8. Compare the retrieval results with the plots. Explain why 2D proximity is not the retrieval method.

## Submission

Commit the completed `lab_8.qmd`, rendered HTML (if requested by your instructor), and both PCA and UMAP figures. Answer the interpretation questions in the QMD. Include your question and the top-three retrieval results.

## Reproducibility and interpretation

PCA and UMAP are fitted by each student. UMAP can vary with package versions and settings, so exact layouts may differ. The relative orientation, rotation, and distances in a two-dimensional visualization should not be overinterpreted. Use cosine similarity on the original vectors for retrieval. UMAP's `transform` places a new query into the fitted visualization; it is not a retrieval algorithm.
