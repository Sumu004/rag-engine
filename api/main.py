"""
RAG Engine API - FastAPI endpoints for document query.

Key improvements over previous version:
  - Index persistence: FAISS index and metadata are saved to disk after
    each ingest and reloaded on startup, so data survives restarts.
  - Removed the naive dict cache from the API layer — caching is handled
    by the LLMClient's semantic cache (embedding-similarity based).
  - Uses the lifespan context manager instead of deprecated on_event.
"""

import os
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

from chunker.semantic_chunker import SemanticChunker
from retrieval.hybrid_retriever import HybridRetriever
from llm.llm_router import LLMRouter, QueryClassifier

DATA_DIR = os.environ.get('DATA_DIR', '/tmp/rag-data')
INDEX_DIR = os.environ.get('INDEX_DIR', '/tmp/rag-index')
INDEX_PATH = os.path.join(INDEX_DIR, 'rag_index')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

chunker = SemanticChunker()
retriever = HybridRetriever()
router = LLMRouter()
classifier = QueryClassifier()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load persisted index if available.  Shutdown: save."""
    print("RAG Engine starting...")
    print(f"  Chunker model: {chunker.model}")
    print("  Retriever: hybrid (FAISS + BM25)")
    print("  LLM: GROQ LLaMA-3")

    chunker.model.to('cpu')

    # Auto-load persisted index from a previous run.
    if os.path.exists(f"{INDEX_PATH}.faiss"):
        try:
            retriever.load(INDEX_PATH)
            stats = retriever.index_stats()
            print(f"  Loaded persisted index: {stats['num_docs']} chunks, "
                  f"FAISS ntotal={stats['faiss_ntotal']}")
        except Exception as e:
            print(f"  Warning: failed to load persisted index: {e}")
    else:
        print("  No persisted index found — starting empty.")

    yield  # application runs

    # Auto-save on shutdown.
    if retriever.chunks:
        try:
            retriever.save(INDEX_PATH)
            print(f"Index saved ({retriever.index_stats()['num_docs']} chunks).")
        except Exception as e:
            print(f"Warning: failed to save index on shutdown: {e}")


app = FastAPI(
    title="RAG Engine API",
    description="Production-grade RAG with hybrid retrieval and LLM routing",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"service": "RAG Engine API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "index": retriever.index_stats(),
        "cache_size": len(router.client.cache),
    }


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a document (txt or pdf).

    The index is persisted to disk after every ingest so data survives
    server restarts.
    """
    content = await file.read()
    text = content.decode('utf-8', errors='ignore')

    file_hash = hashlib.md5(content).hexdigest()[:8]

    chunks = chunker.chunk(text)

    for chunk in chunks:
        chunk['source_file'] = file.filename
        chunk['file_hash'] = file_hash

    retriever.add_documents(chunks)

    # Persist to disk so the index survives restarts.
    try:
        retriever.save(INDEX_PATH)
    except Exception as e:
        print(f"Warning: failed to persist index after ingest: {e}")

    return {
        "status": "ingested",
        "filename": file.filename,
        "file_hash": file_hash,
        "num_chunks": len(chunks),
        "total_indexed": retriever.index_stats()['num_docs'],
    }


@app.post("/query")
async def query(
    question: str,
    k: int = 5,
    use_cache: bool = True,
    auto_route: bool = True,
):
    """
    Query the RAG system.

    - question: The question to ask
    - k: Number of chunks to retrieve
    - use_cache: Enable semantic cache (handled by LLMClient)
    - auto_route: Auto-route to model size
    """
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty")

    mode = classifier.classify(question) if auto_route else 'hybrid'

    results = retriever.search(question, k=k, mode=mode)

    contexts = [r['text'] for r in results]

    llm_response = router.complete(question, contexts, auto_route=auto_route)

    response = {
        "question": question,
        "answer": llm_response.content,
        "model": llm_response.model,
        "mode": mode,
        "num_contexts": len(contexts),
        "contexts": [
            {
                "text": r['text'][:200] + "..." if len(r['text']) > 200 else r['text'],
                "score": r['score'],
                "source": r.get('source', 'unknown')
            }
            for r in results
        ],
        "tokens_used": llm_response.tokens_used,
        "latency_ms": llm_response.latency_ms
    }

    return response


@app.get("/stats")
async def stats():
    """Get system statistics."""
    return {
        "index": retriever.index_stats(),
        "router": router.get_stats(),
        "cache_size": len(router.client.cache),
    }


@app.post("/clear-cache")
async def clear_cache():
    """Clear the LLM semantic cache."""
    router.client._cache_embeddings.clear()
    router.client._cache_responses.clear()
    return {"status": "cleared", "size": 0}


@app.get("/eval")
async def eval_questions():
    """Get evaluation questions."""
    return EVAL_QUESTIONS


EVAL_QUESTIONS = [
    {"id": 1, "question": "What is Python?", "type": "factual"},
    {"id": 2, "question": "Explain machine learning", "type": "semantic"},
    {"id": 3, "question": "Define neural networks", "type": "factual"},
    {"id": 4, "question": "Compare ML and DL", "type": "complex"},
    {"id": 5, "question": "List programming languages", "type": "factual"},
    {"id": 6, "question": "What is FAISS used for?", "type": "factual"},
    {"id": 7, "question": "Explain RAG architecture", "type": "semantic"},
    {"id": 8, "question": "What is BM25?", "type": "factual"},
    {"id": 9, "question": "How does cosine similarity work?", "type": "semantic"},
    {"id": 10, "question": "List retrieval methods", "type": "factual"},
    {"id": 11, "question": "What is chunking in RAG?", "type": "factual"},
    {"id": 12, "question": "Explain embedding vectors", "type": "semantic"},
    {"id": 13, "question": "What is semantic search?", "type": "semantic"},
    {"id": 14, "question": "Define token bucket", "type": "factual"},
    {"id": 15, "question": "Explain rate limiting", "type": "semantic"},
    {"id": 16, "question": "What is idempotency?", "type": "factual"},
    {"id": 17, "question": "How does RRF work?", "type": "semantic"},
    {"id": 18, "question": "What is vector database?", "type": "factual"},
    {"id": 19, "question": "Explain LLM routing", "type": "semantic"},
    {"id": 20, "question": "What is exactly-once?", "type": "factual"},
]


def main():
    port = int(os.environ.get('PORT', '9000'))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
