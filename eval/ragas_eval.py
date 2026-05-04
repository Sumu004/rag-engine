"""
RAGAS evaluation pipeline for CI/CD.

Loads eval/golden_dataset.json, builds a mini RAG pipeline from the corpus,
runs all 20 questions through it, evaluates with RAGAS (faithfulness,
answer_relevancy, context_precision), and exits with code 1 if faithfulness
drops below FAITHFULNESS_THRESHOLD.

Requires GROQ_API_KEY to be set for the LLM judge. Without it, falls back
to a heuristic evaluator that still enforces the threshold — useful for local
dry runs but CI must have the key set.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from chunker.semantic_chunker import SemanticChunker
from retrieval.hybrid_retriever import HybridRetriever
from llm.llm_router import LLMRouter, QueryClassifier

FAITHFULNESS_THRESHOLD = float(os.environ.get("RAGAS_THRESHOLD", "0.85"))
DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# RAGAS setup
# ---------------------------------------------------------------------------

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

RAGAS_LLM_CONFIGURED = False
if RAGAS_AVAILABLE and GROQ_API_KEY:
    try:
        from openai import OpenAI as _OpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import HuggingFaceEmbeddings as _RagasHFEmbeddings

        # GROQ as LLM judge via OpenAI-compatible API
        _groq_client = _OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        _ragas_llm = llm_factory("llama-3-8b-8192", client=_groq_client)

        # HuggingFace embeddings locally — GROQ does not serve embeddings
        _ragas_embeddings = _RagasHFEmbeddings()

        for metric in (faithfulness, answer_relevancy, context_precision):
            metric.llm = _ragas_llm
        # answer_relevancy uses embeddings to score semantic similarity
        answer_relevancy.embeddings = _ragas_embeddings

        RAGAS_LLM_CONFIGURED = True
    except Exception as e:
        print(f"[warn] RAGAS LLM config failed: {e}. Falling back to heuristic.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(corpus: List[Dict]) -> Tuple[HybridRetriever, LLMRouter, QueryClassifier]:
    chunker = SemanticChunker()
    retriever = HybridRetriever()
    router = LLMRouter()
    classifier = QueryClassifier()

    all_chunks = []
    for doc in corpus:
        chunks = chunker.chunk(doc["text"])
        for chunk in chunks:
            chunk["source"] = doc["id"]
        all_chunks.extend(chunks)

    retriever.add_documents(all_chunks)
    return retriever, router, classifier


def run_question(
    question: str,
    retriever: HybridRetriever,
    router: LLMRouter,
    k: int = 5,
) -> Dict:
    mode = retriever.classify_query(question)
    results = retriever.search(question, k=k, mode=mode)
    contexts = [r["text"] for r in results]
    response = router.complete(question, contexts, auto_route=True)
    return {
        "question": question,
        "answer": response.content,
        "contexts": contexts,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_with_ragas(records: List[Dict]) -> Dict:
    """Run RAGAS metrics. Returns dict with metric scores."""
    ds = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in records
    ])
    result = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "evaluator": "ragas",
    }


def evaluate_heuristic(records: List[Dict]) -> Dict:
    """
    Word-overlap proxy for faithfulness when RAGAS/LLM is unavailable.
    Faithfulness proxy: what fraction of answer words appear in the contexts.
    """
    scores = []
    for r in records:
        answer_words = set(r["answer"].lower().split())
        context_words = set(" ".join(r["contexts"]).lower().split())
        if not answer_words:
            scores.append(0.0)
            continue
        overlap = len(answer_words & context_words) / len(answer_words)
        scores.append(overlap)

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "faithfulness": avg,
        "answer_relevancy": avg,
        "context_precision": avg,
        "evaluator": "heuristic",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("RAGAS Evaluation Pipeline")
    print("=" * 60)

    if not DATASET_PATH.exists():
        print(f"[error] Golden dataset not found: {DATASET_PATH}", file=sys.stderr)
        return 1

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    corpus = dataset["corpus"]
    qa_pairs = dataset["qa_pairs"]

    print(f"Corpus: {len(corpus)} documents")
    print(f"Questions: {len(qa_pairs)}")

    print("\nBuilding RAG pipeline...")
    retriever, router, _ = build_pipeline(corpus)
    stats = retriever.index_stats()
    print(f"Index: {stats['num_docs']} chunks, dim={stats['embedding_dim']}")

    print("\nRunning questions through pipeline...")
    records = []
    for i, qa in enumerate(qa_pairs, 1):
        print(f"  [{i:02d}/{len(qa_pairs)}] {qa['question'][:60]}")
        result = run_question(qa["question"], retriever, router)
        result["ground_truth"] = qa["ground_truth"]
        records.append(result)

    print("\nEvaluating...")
    if RAGAS_AVAILABLE and RAGAS_LLM_CONFIGURED:
        scores = evaluate_with_ragas(records)
    else:
        if RAGAS_AVAILABLE and not RAGAS_LLM_CONFIGURED:
            print("[warn] GROQ_API_KEY not set — using heuristic evaluator", file=sys.stderr)
        elif not RAGAS_AVAILABLE:
            print("[warn] ragas not installed — using heuristic evaluator", file=sys.stderr)
        scores = evaluate_heuristic(records)

    print(f"\nResults (evaluator: {scores['evaluator']}):")
    print(f"  Faithfulness:      {scores['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']:.3f}")
    print(f"  Context Precision: {scores['context_precision']:.3f}")
    print(f"  Threshold:         {FAITHFULNESS_THRESHOLD:.3f}")

    passed = scores["faithfulness"] >= FAITHFULNESS_THRESHOLD
    print(f"  Faithfulness gate: {'PASSED ✓' if passed else 'FAILED ✗'}")

    if not passed:
        print(
            f"\n[error] Faithfulness {scores['faithfulness']:.3f} < threshold {FAITHFULNESS_THRESHOLD:.3f}. "
            "Build failed.",
            file=sys.stderr,
        )
        return 1

    print("\nEvaluation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
