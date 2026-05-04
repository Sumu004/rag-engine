"""
RAG evaluation pipeline for CI/CD.

Loads eval/golden_dataset.json, builds a mini RAG pipeline from the corpus,
runs all 20 questions through it, then evaluates faithfulness using a direct
GROQ LLM judge (same concept as RAGAS, zero external dependency on RAGAS
internals). Exits with code 1 if faithfulness drops below FAITHFULNESS_THRESHOLD.

GROQ_API_KEY must be set in CI (GitHub Actions secret). Without it, the
pipeline runs but skips the LLM judge and uses a word-overlap heuristic.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from chunker.semantic_chunker import SemanticChunker
from retrieval.hybrid_retriever import HybridRetriever
from llm.llm_router import LLMRouter

FAITHFULNESS_THRESHOLD = float(os.environ.get("RAGAS_THRESHOLD", "0.85"))
DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "llama-3.1-8b-instant"

FAITHFULNESS_PROMPT = """\
You are an evaluation judge assessing whether an answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Is the answer grounded in and supported by the context above?
Respond with a single decimal number between 0.0 (not faithful) and 1.0 (fully faithful). No explanation."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(corpus: List[Dict]) -> Tuple[HybridRetriever, LLMRouter]:
    chunker = SemanticChunker()
    retriever = HybridRetriever()
    router = LLMRouter()

    all_chunks = []
    for doc in corpus:
        chunks = chunker.chunk(doc["text"])
        for chunk in chunks:
            chunk["source"] = doc["id"]
        all_chunks.extend(chunks)

    retriever.add_documents(all_chunks)
    return retriever, router


def run_question(question: str, retriever: HybridRetriever, router: LLMRouter, k: int = 5) -> Dict:
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

def _groq_chat(prompt: str, retries: int = 4) -> str:
    """Single GROQ completion with exponential backoff on 429."""
    delay = 2.0
    for attempt in range(retries):
        resp = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 10,
            },
            timeout=30,
        )
        if resp.status_code == 429:
            wait = delay * (2 ** attempt)
            print(f"  [rate-limit] 429 — retrying in {wait:.0f}s (attempt {attempt + 1}/{retries})", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    resp.raise_for_status()  # raise after exhausting retries
    return ""  # unreachable


def evaluate_with_llm_judge(records: List[Dict]) -> Dict:
    """
    LLM-judge faithfulness: ask GROQ to score each answer 0–1 against its contexts.
    Rate-limit: 0.5s between requests to stay within GROQ free-tier limits.
    """
    scores = []
    for r in records:
        context = "\n\n".join(r["contexts"][:3])
        prompt = FAITHFULNESS_PROMPT.format(context=context, answer=r["answer"])
        try:
            content = _groq_chat(prompt)
            score = float(content.split()[0])
            scores.append(min(1.0, max(0.0, score)))
        except Exception as e:
            print(f"  [warn] Judge failed for '{r['question'][:40]}': {e}", file=sys.stderr)
            scores.append(0.5)
        time.sleep(2.0)

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "faithfulness": avg,
        "evaluator": "llm-judge-groq",
        "num_evaluated": len(scores),
    }


def evaluate_heuristic(records: List[Dict]) -> Dict:
    """Word-overlap proxy — used only when GROQ_API_KEY is not set."""
    scores = []
    for r in records:
        answer_words = set(r["answer"].lower().split())
        context_words = set(" ".join(r["contexts"]).lower().split())
        overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0.0
        scores.append(overlap)
    avg = sum(scores) / len(scores) if scores else 0.0
    return {"faithfulness": avg, "evaluator": "heuristic", "num_evaluated": len(scores)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("RAG Evaluation Pipeline")
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
    retriever, router = build_pipeline(corpus)
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
    if GROQ_API_KEY:
        print(f"  Judge: {JUDGE_MODEL} via GROQ")
        scores = evaluate_with_llm_judge(records)
    else:
        print("[warn] GROQ_API_KEY not set — using word-overlap heuristic", file=sys.stderr)
        scores = evaluate_heuristic(records)

    print(f"\nResults (evaluator: {scores['evaluator']}):")
    print(f"  Faithfulness:  {scores['faithfulness']:.3f}")
    print(f"  Threshold:     {FAITHFULNESS_THRESHOLD:.3f}")
    print(f"  Questions:     {scores['num_evaluated']}")

    passed = scores["faithfulness"] >= FAITHFULNESS_THRESHOLD
    print(f"  Gate:          {'PASSED ✓' if passed else 'FAILED ✗'}")

    if not passed:
        print(
            f"\n[error] Faithfulness {scores['faithfulness']:.3f} < threshold "
            f"{FAITHFULNESS_THRESHOLD:.3f}. Build failed.",
            file=sys.stderr,
        )
        return 1

    print("\nEvaluation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
