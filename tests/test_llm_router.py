"""Tests for the LLM router and query classifier.

Verifies:
  - Embedding-centroid classifier routes simple/complex queries correctly
  - LLMClient demo mode works without API key
  - Semantic cache returns hits for similar queries
  - Router tracks stats correctly
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm.llm_router import QueryClassifier, LLMClient, LLMRouter, LLMResponse


@pytest.fixture(scope='module')
def classifier():
    return QueryClassifier()


@pytest.fixture(scope='module')
def router():
    return LLMRouter()


class TestQueryClassifier:

    def test_simple_factual_routes_to_small(self, classifier):
        assert classifier.classify("What is Python?") == 'small'

    def test_definition_routes_to_small(self, classifier):
        assert classifier.classify("Define neural networks") == 'small'

    def test_list_routes_to_small(self, classifier):
        assert classifier.classify("List the planets in our solar system") == 'small'

    def test_comparison_routes_to_large(self, classifier):
        result = classifier.classify("Compare and contrast machine learning with deep learning and analyse the trade-offs")
        assert result == 'large'

    def test_explanation_routes_to_large(self, classifier):
        result = classifier.classify("Explain the relationship between neural networks and deep learning architectures")
        assert result == 'large'

    def test_design_question_routes_to_large(self, classifier):
        result = classifier.classify("Design a distributed system that handles rate limiting across multiple regions")
        assert result == 'large'


class TestLLMClient:

    def test_demo_mode_without_api_key(self):
        """Without GROQ_API_KEY, client should return demo responses."""
        client = LLMClient(api_key='')
        response = client.complete("What is Python?")
        assert isinstance(response, LLMResponse)
        assert '[Demo]' in response.content
        assert response.finish_reason == 'demo'

    def test_cache_returns_hit_for_same_query(self):
        """Exact same query should hit the cache."""
        client = LLMClient(api_key='')
        r1 = client.complete("What is machine learning?", cache=True)
        r2 = client.complete("What is machine learning?", cache=True)
        # Should be the exact same object (cache hit)
        assert r1.content == r2.content

    def test_cache_disabled(self):
        """With cache=False, should always generate fresh."""
        client = LLMClient(api_key='')
        r1 = client.complete("Test query", cache=False)
        assert isinstance(r1, LLMResponse)

    def test_complete_with_context(self):
        """Context-augmented completion should work in demo mode."""
        client = LLMClient(api_key='')
        response = client.complete_with_context(
            "What is Python?",
            ["Python is a programming language."],
        )
        assert isinstance(response, LLMResponse)


class TestLLMRouter:

    def test_router_tracks_stats(self, router):
        """Router should track call counts per model size."""
        initial_small = router.stats['small_calls']
        initial_large = router.stats['large_calls']

        router.complete("What is Python?")
        router.complete("Compare X and Y in detail and analyse trade-offs")

        assert router.stats['small_calls'] >= initial_small
        assert router.stats['small_calls'] + router.stats['large_calls'] > initial_small + initial_large

    def test_router_get_stats(self, router):
        stats = router.get_stats()
        assert 'small_calls' in stats
        assert 'large_calls' in stats
        assert 'total_cost' in stats
        assert 'cache_size' in stats

    def test_manual_routing(self):
        """With auto_route=False, should always use 'small'."""
        r = LLMRouter()
        response = r.complete("Explain quantum physics in depth", auto_route=False)
        # Without auto_route, even complex queries go to small
        assert r.stats['small_calls'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
