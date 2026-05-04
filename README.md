# Adaptive RAG Engine

Production-grade RAG API with hybrid retrieval and LLM routing.

## Architecture

```
Document → Semantic Chunker → FAISS + BM25 → LLM (GROQ LLaMA-3) → Answer
                         ↑                                    ↓
                    Reciprocal Rank Fusion         Semantic Cache
```

## Features

- **Semantic Chunker**: Splits by cosine similarity between sentence embeddings
- **Hybrid Retrieval**: FAISS (dense) + BM25 (sparse) with RRF
- **Query Classification**: Route to dense, sparse, or hybrid based on query type
- **LLM Routing**: LLaMA-3-8B (simple) / LLaMA-3-70B (complex)
- **Semantic Cache**: Cache similar queries
- **RAGAS Evaluation**: Automated quality metrics in CI/CD

## Tech Stack

- FastAPI
- FAISS + BM25
- sentence-transformers
- GROQ (LLaMA-3)
- RAGAS

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests
pytest tests/test_rag.py -v

# Start API
python api/main.py
```

## API Usage

```bash
# Ingest document
curl -X POST http://localhost:9000/ingest \
  -F "file=@document.txt"

# Query
curl -X POST http://localhost:9000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Get stats
curl http://localhost:9000/stats
```

## Evaluation

```bash
# Run RAGAS eval
python eval/ragas_eval.py

# Get eval questions
curl http://localhost:9000/eval
```

## Project Structure

```
rag-engine/
├── api/              # FastAPI endpoints
├── chunker/           # Semantic chunker
├── retrieval/          # Hybrid FAISS + BM25
├── llm/              # LLM router
├── eval/              # RAGAS evaluation
└── tests/            # Unit tests
```

## License

MIT