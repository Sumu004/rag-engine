# Adaptive RAG Engine

Production-grade Retrieval-Augmented Generation API with hybrid retrieval, semantic caching, and LLM routing.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              RAG Engine                       │
                    │                                              │
  Document ────────►│  ┌─────────────────┐                         │
  (POST /ingest)    │  │ Semantic Chunker │                         │
                    │  │ (cosine splits   │                         │
                    │  │  + overlap)      │                         │
                    │  └────────┬────────┘                         │
                    │           │                                   │
                    │     ┌─────┴──────┐                           │
                    │     │            │                            │
                    │     ▼            ▼                            │
                    │  ┌──────┐  ┌─────────┐                       │
                    │  │FAISS │  │  BM25   │                       │
                    │  │(dense│  │ (sparse)│                       │
                    │  │index)│  │         │                       │
                    │  └──┬───┘  └────┬────┘                       │
                    │     │           │                             │
  Query ───────────►│     └─────┬─────┘                            │
  (POST /query)     │           │                                  │
                    │    ┌──────┴───────┐                           │
                    │    │  Reciprocal  │                           │
                    │    │  Rank Fusion │                           │
                    │    └──────┬───────┘                           │
                    │           │                                   │
                    │    ┌──────┴───────┐    ┌────────────────┐    │
                    │    │   Query      │    │  Semantic      │    │
                    │    │  Classifier  │◄──►│  Cache (FAISS) │    │
                    │    │  (embedding  │    │  + LRU evict   │    │
                    │    │   centroid)  │    └────────────────┘    │
                    │    └──────┬───────┘                           │
                    │           │                                   │
                    │     ┌─────┴─────┐                            │
                    │     │           │                             │
                    │     ▼           ▼                             │
                    │  ┌───────┐  ┌───────┐                        │
                    │  │LLaMA-3│  │LLaMA-3│                        │
                    │  │ 8B    │  │ 70B   │                        │
                    │  │(fast) │  │(deep) │                        │
                    │  └───┬───┘  └───┬───┘                        │
                    │      └────┬─────┘                            │
                    │           ▼                                   │
                    │      ┌─────────┐     ┌──────────────────┐   │
                    │      │ Answer  │     │ Index Persistence │   │
                    │      └─────────┘     │ (FAISS + JSON)   │   │
                    │                      └──────────────────┘   │
                    └──────────────────────────────────────────────┘
```

## Design Decisions

### Why semantic chunking with overlap?
Fixed-token chunking blindly splits mid-sentence. Semantic chunking splits where cosine similarity between adjacent sentence embeddings drops below a threshold, preserving topical boundaries. A configurable sentence overlap (default: 2) between chunks ensures context isn't lost at boundaries — critical for questions that span chunk edges.

### Why hybrid retrieval (FAISS + BM25)?
Dense retrieval (FAISS) captures semantic similarity — "programming language" matches "Python" even without exact keyword overlap. Sparse retrieval (BM25) excels at exact matches — "BM25 Okapi" finds the right document instantly. Reciprocal Rank Fusion (RRF) combines both without score normalisation, giving the best of both worlds.

### Why incremental indexing?
The previous implementation rebuilt the entire FAISS index on every ingest, deleting all previously indexed documents. The current implementation appends new embeddings to the existing index in-place. BM25 must still be rebuilt (IDF statistics are global) but the FAISS index grows incrementally.

### Why embedding-centroid query classification?
Keyword lists ("what is" → simple, "explain" → complex) are fragile and incomplete. The classifier pre-computes centroid embeddings from archetypal simple and complex queries, then classifies new queries by cosine distance. This is more robust and requires zero maintenance.

### Why FAISS-backed semantic cache?
The previous cache was a Python dict with string keys (no semantic matching) and no eviction (unbounded growth). The current cache uses a FAISS index for O(1)-ish similarity lookup and an OrderedDict for LRU eviction at a configurable capacity (default: 1000).

### Why auto-persist the index?
The index is saved to disk after every ingest and loaded on startup. Without this, all indexed documents were lost on server restart — a showstopper for any real deployment.

## Features

- **Semantic Chunker** — cosine-similarity splits + configurable sentence overlap
- **Hybrid Retrieval** — FAISS (dense) + BM25 (sparse) with Reciprocal Rank Fusion
- **Embedding-Centroid Query Classification** — routes to dense, sparse, or hybrid
- **LLM Routing** — simple queries → LLaMA-3-8B (fast), complex → LLaMA-3-70B (deep)
- **FAISS Semantic Cache** — embedding similarity lookup with LRU eviction
- **Index Persistence** — auto-save on ingest, auto-load on startup
- **Incremental Indexing** — ingest new documents without rebuilding
- **RAGAS Evaluation** — faithfulness quality gate in CI/CD
- **Multi-Metric Evaluation** — faithfulness + word-overlap heuristic

## Tech Stack

- FastAPI (async)
- FAISS (dense vector search)
- BM25 via `rank_bm25` (sparse retrieval)
- `sentence-transformers` (all-MiniLM-L6-v2)
- GROQ (LLaMA-3-8B + LLaMA-3-70B)
- GitHub Actions (CI with quality gate)

## Project Structure

```
rag-engine/
├── api/
│   └── main.py                # FastAPI endpoints + auto-persistence
├── chunker/
│   └── semantic_chunker.py    # Cosine-split chunking + overlap
├── retrieval/
│   └── hybrid_retriever.py    # Incremental FAISS + BM25 + RRF
├── llm/
│   └── llm_router.py          # Centroid classifier + FAISS cache + router
├── eval/
│   ├── ragas_eval.py           # Faithfulness evaluation pipeline
│   └── golden_dataset.json     # 20 QA pairs with ground truth
├── tests/
│   ├── test_chunker.py         # Semantic chunking tests
│   ├── test_retriever.py       # Retrieval + incremental indexing tests
│   ├── test_llm_router.py      # Classifier + cache + routing tests
│   └── test_rag.py             # Basic smoke tests
└── .github/workflows/
    └── ragas_eval.yml          # CI quality gate
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start API
python api/main.py
```

## API Usage

```bash
# Ingest document
curl -X POST http://localhost:9000/ingest \
  -F "file=@document.txt"

# Query (auto-routes to appropriate model)
curl -X POST http://localhost:9000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Get stats
curl http://localhost:9000/stats

# Clear cache
curl -X POST http://localhost:9000/clear-cache
```

## Evaluation

```bash
# Run RAGAS eval (requires GROQ_API_KEY)
GROQ_API_KEY=your-key python eval/ragas_eval.py

# Without API key (uses word-overlap heuristic)
python eval/ragas_eval.py
```

## License

MIT