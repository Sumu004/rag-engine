"""
RAG Engine API - FastAPI endpoints for document query.
"""

import os
import sys
import json
import hashlib
from typing import List, Optional
from dataclasses import dataclass

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from chunker.semantic_chunker import SemanticChunker
from retrieval.hybrid_retriever import HybridRetriever
from llm.llm_router import LLMRouter, QueryClassifier

app = FastAPI(
    title="RAG Engine API",
    description="Production-grade RAG with hybrid retrieval and LLM routing",
    version="1.0.0"
)

DATA_DIR = os.environ.get('DATA_DIR', '/tmp/rag-data')
INDEX_DIR = os.environ.get('INDEX_DIR', '/tmp/rag-index')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

chunker = SemanticChunker()
retriever = HybridRetriever()
router = LLMRouter()
classifier = QueryClassifier()

query_cache: dict = {}


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    print("RAG Engine starting...")
    print(f"  Chunker model: {chunker.model}")
    print(f"  Retriever: hybrid (FAISS + BM25)")
    print(f"  LLM: GROQ LLaMA-3")
    
    chunker.model.to('cpu')


@app.get("/")
async def root():
    return {"service": "RAG Engine API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "index": retriever.index_stats(),
        "cache_size": len(query_cache)
    }


@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a document (txt or pdf)."""
    content = await file.read()
    text = content.decode('utf-8', errors='ignore')
    
    file_hash = hashlib.md5(content).hexdigest()[:8]
    
    chunks = chunker.chunk(text)
    
    for chunk in chunks:
        chunk['source_file'] = file.filename
        chunk['file_hash'] = file_hash
    
    retriever.add_documents(chunks)
    
    return {
        "status": "ingested",
        "filename": file.filename,
        "file_hash": file_hash,
        "num_chunks": len(chunks)
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
    - use_cache: Enable semantic cache
    - auto_route: Auto-route to model size
    """
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty")
    
    if use_cache:
        cache_key = f"{question}:{k}"
        if cache_key in query_cache:
            return query_cache[cache_key]
    
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
    
    if use_cache:
        query_cache[cache_key] = response
    
    return response


@app.get("/stats")
async def stats():
    """Get system statistics."""
    return {
        "index": retriever.index_stats(),
        "router": router.get_stats(),
        "cache_size": len(query_cache)
    }


@app.post("/clear-cache")
async def clear_cache():
    """Clear query cache."""
    query_cache.clear()
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