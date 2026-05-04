"""
LLM Router - Routes queries to different model sizes based on complexity.
Uses LLaMA-3 via GROQ API for fast inference.
"""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_BASE_URL = 'https://api.groq.com/openai/v1'

ModelSize = Enum('ModelSize', 'SMALL LARGE')


LLM_MODELS = {
    'small': 'llama-3.1-8b-instant',
    'large': 'llama-3.3-70b-versatile',
}


class QueryClassifier:
    """Classifies query complexity for routing."""
    
    @staticmethod
    def classify(query: str) -> str:
        """
        Classify query complexity.
        Returns 'small' for simple, 'large' for complex.
        """
        query_lower = query.lower()
        
        simple_indicators = [
            'what is', 'list', 'define', 'name',
            'how many', 'when', 'who',
            'boolean', 'true or false',
        ]
        
        complex_indicators = [
            'explain', 'compare', 'analyze',
            'relationship', 'why', 'differences',
            'step by step', 'pros and cons',
            'evaluate', 'design',
        ]
        
        simple_score = sum(1 for ind in simple_indicators if ind in query_lower)
        complex_score = sum(1 for ind in complex_indicators if ind in query_lower)
        
        word_count = len(query.split())
        if word_count > 30:
            complex_score += 2
        elif word_count > 15:
            complex_score += 1
        
        entity_count = query.count('?') + query.count(' and ')
        if entity_count > 2:
            complex_score += 1
        
        if complex_score > simple_score:
            return 'large'
        else:
            return 'small'


@dataclass
class LLMResponse:
    """LLM response container."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    finish_reason: str


_EMBED_MODEL = None

def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))
    return _EMBED_MODEL


class LLMClient:
    """GROQ LLaMA-3 client with semantic cache."""

    def __init__(self, api_key: str = GROQ_API_KEY):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY', '')
        self.base_url = GROQ_BASE_URL
        self.cache_threshold = float(os.environ.get('CACHE_THRESHOLD', '0.92'))
        # Semantic cache: list of (embedding, LLMResponse) pairs.
        self._cache_embeddings: List[np.ndarray] = []
        self._cache_responses: List[LLMResponse] = []

    @property
    def cache(self) -> Dict:
        """Read-only size proxy used by LLMRouter.get_stats()."""
        return {i: None for i in range(len(self._cache_responses))}

    def _embed(self, text: str) -> np.ndarray:
        vec = _get_embed_model().encode([text], convert_to_numpy=True)[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _cache_lookup(self, prompt_vec: np.ndarray) -> Optional[LLMResponse]:
        """Return a cached response if any stored embedding exceeds threshold."""
        if not self._cache_embeddings:
            return None
        matrix = np.stack(self._cache_embeddings)          # (N, dim)
        similarities = matrix @ prompt_vec                  # cosine (pre-normalised)
        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] >= self.cache_threshold:
            return self._cache_responses[best_idx]
        return None

    def _cache_store(self, prompt_vec: np.ndarray, response: LLMResponse) -> None:
        self._cache_embeddings.append(prompt_vec)
        self._cache_responses.append(response)

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
        prompt_vec = self._embed(prompt) if cache else None

        if cache:
            hit = self._cache_lookup(prompt_vec)
            if hit is not None:
                return hit
        
        if not self.api_key:
            demo = LLMResponse(
                content=f"[Demo] Would answer: {prompt[:100]}... (GROQ_API_KEY not set)",
                model=model,
                tokens_used=0,
                latency_ms=0,
                finish_reason='demo'
            )
            if cache:
                self._cache_store(prompt_vec, demo)
            return demo
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': LLM_MODELS.get(model, LLM_MODELS['small']),
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        try:
            import time
            start = time.time()
            
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            latency = (time.time() - start) * 1000
            
            if response.status_code != 200:
                return LLMResponse(
                    content=f"Error: {response.status_code} - {response.text[:100]}",
                    model=model,
                    tokens_used=0,
                    latency_ms=latency,
                    finish_reason='error'
                )
            
            data = response.json()
            choice = data.get('choices', [{}])[0]
            message = choice.get('message', {})
            
            result = LLMResponse(
                content=message.get('content', ''),
                model=model,
                tokens_used=data.get('usage', {}).get('total_tokens', 0),
                latency_ms=latency,
                finish_reason=choice.get('finish_reason', '')
            )
            
            if cache:
                self._cache_store(prompt_vec, result)

            return result
            
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)[:100]}",
                model=model,
                tokens_used=0,
                latency_ms=0,
                finish_reason='error'
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
    
    def complete(self, query: str, context: List[str] = None, auto_route: bool = True) -> LLMResponse:
        """
        Complete with optional auto-routing.
        If auto_route=True, classify query complexity and route to appropriate model.
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
        "Compare machine learning vs deep learning",
    ]
    
    print("\nQuery Classification:")
    for query in tests:
        size = classifier.classify(query)
        print(f"  '{query[:40]}...' → {size.upper()}")
    
    print("\n" + "=" * 50)
    print("Router stats available via router.get_stats()")
    print("=" * 50)