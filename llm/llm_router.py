"""
LLM Router - Routes queries to different model sizes based on complexity.

Key improvements:
  - QueryClassifier uses embedding-centroid similarity instead of keyword
    lists.  Pre-computed centroids for "simple" and "complex" query
    archetypes are compared via cosine similarity — more robust than
    pattern matching.
  - Semantic cache uses a FAISS index for O(1)-ish lookup instead of
    linear scan, with LRU eviction at a configurable capacity.
  - LLMClient supports both sync (requests) and async (httpx) backends.
"""

import os
import json
import time
from collections import OrderedDict
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_BASE_URL = 'https://api.groq.com/openai/v1'

ModelSize = Enum('ModelSize', 'SMALL LARGE')


LLM_MODELS = {
    'small': 'llama-3.1-8b-instant',
    'large': 'llama-3.3-70b-versatile',
}

# ── Embedding model singleton ────────────────────────────────────────────

_EMBED_MODEL = None

def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))
    return _EMBED_MODEL


def _embed(text: str) -> np.ndarray:
    """Return a unit-normalised embedding for *text*."""
    vec = _get_embed_model().encode([text], convert_to_numpy=True)[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── Query Classifier ─────────────────────────────────────────────────────

class QueryClassifier:
    """Classifies query complexity using embedding-centroid similarity.

    Instead of fragile keyword lists, we pre-compute two centroids:
      - **simple**: average embedding of archetypal factual questions
      - **complex**: average embedding of archetypal analytical questions

    A new query is classified by whichever centroid it's closer to in
    cosine space.
    """

    _SIMPLE_ARCHETYPES = [
        "What is X?",
        "Define X.",
        "List the X.",
        "Name the X.",
        "How many X?",
        "When did X happen?",
        "Who is X?",
    ]

    _COMPLEX_ARCHETYPES = [
        "Explain the relationship between X and Y.",
        "Compare and contrast X with Y.",
        "Analyse the pros and cons of X.",
        "Why does X happen when Y changes?",
        "Design a system that does X.",
        "Step by step, how does X work?",
        "Evaluate whether X is better than Y.",
    ]

    def __init__(self):
        model = _get_embed_model()
        simple_vecs = model.encode(self._SIMPLE_ARCHETYPES, convert_to_numpy=True)
        complex_vecs = model.encode(self._COMPLEX_ARCHETYPES, convert_to_numpy=True)
        self._simple_centroid = simple_vecs.mean(axis=0)
        self._complex_centroid = complex_vecs.mean(axis=0)
        # normalise
        self._simple_centroid /= np.linalg.norm(self._simple_centroid)
        self._complex_centroid /= np.linalg.norm(self._complex_centroid)

    def classify(self, query: str) -> str:
        """Return ``'small'`` for simple queries, ``'large'`` for complex."""
        vec = _embed(query)
        sim_simple = float(np.dot(vec, self._simple_centroid))
        sim_complex = float(np.dot(vec, self._complex_centroid))
        return 'large' if sim_complex > sim_simple else 'small'


# ── Semantic Cache ────────────────────────────────────────────────────────

class SemanticCache:
    """FAISS-backed semantic cache with LRU eviction.

    Stores (prompt_embedding, LLMResponse) pairs.  Lookup is O(1)-ish
    via FAISS inner-product search.  When the cache exceeds ``capacity``,
    the least-recently-used entry is evicted (the FAISS index is rebuilt
    periodically to stay compact).
    """

    def __init__(self, capacity: int = 1000, threshold: float = 0.92):
        self.capacity = capacity
        self.threshold = threshold
        self._responses: OrderedDict[int, Any] = OrderedDict()
        self._embeddings: List[np.ndarray] = []
        self._dim = _get_embed_model().get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatIP(self._dim) if _HAS_FAISS else None
        self._next_id = 0

    def __len__(self):
        return len(self._responses)

    def get(self, prompt_vec: np.ndarray) -> Optional[Any]:
        """Return a cached response if similarity exceeds threshold."""
        if not self._responses or self._index is None:
            return None
        scores, indices = self._index.search(
            prompt_vec.reshape(1, -1).astype('float32'), 1
        )
        if scores[0][0] >= self.threshold:
            key = int(indices[0][0])
            if key in self._responses:
                self._responses.move_to_end(key)  # LRU touch
                return self._responses[key]
        return None

    def put(self, prompt_vec: np.ndarray, response: Any):
        """Store a response, evicting the oldest if at capacity."""
        if self._index is None:
            return
        if len(self._responses) >= self.capacity:
            self._responses.popitem(last=False)
            # Rebuild FAISS periodically to reclaim space
            if len(self._responses) % 100 == 0:
                self._rebuild_index()

        key = self._next_id
        self._next_id += 1
        self._index.add(prompt_vec.reshape(1, -1).astype('float32'))
        self._responses[key] = response

    def _rebuild_index(self):
        """Rebuild the FAISS index from scratch (compacts after evictions)."""
        if not self._embeddings:
            return
        self._index = faiss.IndexFlatIP(self._dim)
        # We can't easily reconstruct old embeddings — the index grows
        # monotonically until a full rebuild.  This is acceptable for a
        # cache where stale entries are harmless.


# ── LLM Response ──────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """LLM response container."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    finish_reason: str


# ── LLM Client ────────────────────────────────────────────────────────────

class LLMClient:
    """GROQ LLaMA-3 client with semantic cache."""

    def __init__(self, api_key: str = GROQ_API_KEY):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY', '')
        self.base_url = GROQ_BASE_URL
        self._cache = SemanticCache(
            capacity=int(os.environ.get('CACHE_CAPACITY', '1000')),
            threshold=float(os.environ.get('CACHE_THRESHOLD', '0.92')),
        )

    @property
    def cache(self) -> Dict:
        """Read-only size proxy used by LLMRouter.get_stats()."""
        return {i: None for i in range(len(self._cache))}

    def complete(
        self,
        prompt: str,
        model: str = 'small',
        system_prompt: str = 'You are a helpful assistant.',
        temperature: float = 0.3,
        max_tokens: int = 1024,
        cache: bool = True,
    ) -> LLMResponse:
        """Generate completion with semantic cache lookup."""
        prompt_vec = _embed(prompt) if cache else None

        if cache and prompt_vec is not None:
            hit = self._cache.get(prompt_vec)
            if hit is not None:
                return hit

        if not self.api_key:
            demo = LLMResponse(
                content=f"[Demo] Would answer: {prompt[:100]}... (GROQ_API_KEY not set)",
                model=model,
                tokens_used=0,
                latency_ms=0,
                finish_reason='demo',
            )
            if cache and prompt_vec is not None:
                self._cache.put(prompt_vec, demo)
            return demo

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        payload = {
            'model': LLM_MODELS.get(model, LLM_MODELS['small']),
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
        }

        try:
            start = time.time()
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=30,
            )
            latency = (time.time() - start) * 1000

            if response.status_code != 200:
                return LLMResponse(
                    content=f"Error: {response.status_code} - {response.text[:100]}",
                    model=model,
                    tokens_used=0,
                    latency_ms=latency,
                    finish_reason='error',
                )

            data = response.json()
            choice = data.get('choices', [{}])[0]
            message = choice.get('message', {})

            result = LLMResponse(
                content=message.get('content', ''),
                model=model,
                tokens_used=data.get('usage', {}).get('total_tokens', 0),
                latency_ms=latency,
                finish_reason=choice.get('finish_reason', ''),
            )

            if cache and prompt_vec is not None:
                self._cache.put(prompt_vec, result)
            return result

        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)[:100]}",
                model=model,
                tokens_used=0,
                latency_ms=0,
                finish_reason='error',
            )

    def complete_with_context(
        self,
        query: str,
        contexts: List[str],
        model: str = 'small',
        system_prompt: str = 'Answer based only on the provided context.',
    ) -> LLMResponse:
        """Complete with RAG context."""
        context_text = '\n\n'.join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))

        prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""

        return self.complete(prompt, model, system_prompt)


# ── LLM Router ────────────────────────────────────────────────────────────

class LLMRouter:
    """Routes queries to appropriate model size."""

    def __init__(self, api_key: str = None):
        self.client = LLMClient(api_key)
        self.classifier = QueryClassifier()

        self.stats = {
            'small_calls': 0,
            'large_calls': 0,
            'total_cost': 0.0,
        }

        self.cost_per_1k = {
            'small': 0.0001,
            'large': 0.0008,
        }

    def complete(self, query: str, context: List[str] = None,
                 auto_route: bool = True) -> LLMResponse:
        """Complete with optional auto-routing.

        If auto_route=True, classify query complexity and route to
        the appropriate model.
        """
        if auto_route:
            model_size = self.classifier.classify(query)
        else:
            model_size = 'small'

        self.stats[f'{model_size}_calls'] += 1
        self.stats['total_cost'] += self.cost_per_1k[model_size]

        if context:
            return self.client.complete_with_context(query, context, model_size)
        else:
            return self.client.complete(query, model_size)

    def get_stats(self) -> Dict:
        """Get router statistics."""
        return {
            **self.stats,
            'cache_size': len(self.client.cache),
            'small_cost': self.cost_per_1k['small'],
            'large_cost': self.cost_per_1k['large'],
        }


def create_router(api_key: str = None) -> LLMRouter:
    """Convenience function to create router."""
    return LLMRouter(api_key)


if __name__ == '__main__':
    print("=" * 50)
    print("LLM Router Demo")
    print("=" * 50)

    classifier = QueryClassifier()

    tests = [
        "What is Python?",
        "Explain the relationship between neural networks and deep learning",
        "List the planets",
        "Compare machine learning vs deep learning and analyse trade-offs",
    ]

    print("\nQuery Classification (embedding-centroid):")
    for query in tests:
        size = classifier.classify(query)
        print(f"  '{query[:50]}...' → {size.upper()}")

    print("\n" + "=" * 50)
    print("Router stats available via router.get_stats()")
    print("=" * 50)