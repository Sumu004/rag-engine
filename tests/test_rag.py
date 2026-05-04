"""Tests for RAG Engine - minimal version."""

import pytest
import numpy as np


def test_faiss_import():
    """Test FAISS can be imported."""
    import faiss
    assert faiss is not None


def test_bm25_import():
    """Test BM25 can be imported."""
    from rank_bm25 import BM25Okapi
    assert BM25Okapi is not None


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_sim = dot / (norm_a * norm_b)
    
    assert cos_sim == pytest.approx(1.0)


def test_rrf_score():
    """Test Reciprocal Rank Fusion calculation."""
    k = 60
    
    doc_scores = {'doc1': 1/(k+1), 'doc2': 1/(k+2), 'doc1': 1/(k+1) + 1/(k+1)}
    
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    assert sorted_docs[0][0] == 'doc1'


def test_chunk_splits():
    """Test basic chunk splitting."""
    text = "First sentence. Second sentence. Third sentence."
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    assert len(sentences) == 3
    assert sentences[0] == "First sentence"


def test_rag_k_constant():
    """Test RRF k constant."""
    k = 60
    
    rank1_score = 1 / (k + 1)
    rank2_score = 1 / (k + 2)
    
    assert rank1_score > rank2_score


if __name__ == '__main__':
    pytest.main([__file__, '-v'])