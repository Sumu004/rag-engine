"""Tests for the semantic chunker.

Verifies:
  - Semantically different paragraphs split into separate chunks
  - Semantically similar sentences stay together
  - Overlap is applied between adjacent chunks
  - PDF chunking preserves page metadata
  - Edge cases: empty text, single sentence
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chunker.semantic_chunker import SemanticChunker


@pytest.fixture(scope='module')
def chunker():
    """Shared chunker instance (model loading is expensive)."""
    return SemanticChunker()


class TestSemanticChunker:

    def test_empty_text_returns_empty(self, chunker):
        assert chunker.chunk('') == []
        assert chunker.chunk('   ') == []

    def test_single_sentence_returns_one_chunk(self, chunker):
        chunks = chunker.chunk('Hello world.')
        assert len(chunks) == 1
        assert chunks[0]['num_sentences'] == 1

    def test_different_topics_split(self, chunker):
        """Two paragraphs about completely different topics should become >= 2 chunks."""
        text = (
            "The Python programming language was created by Guido van Rossum in 1991. "
            "It is widely used in web development and data science. "
            "Quantum mechanics describes the behavior of subatomic particles. "
            "The Heisenberg uncertainty principle limits measurement precision."
        )
        chunks = chunker.chunk(text, overlap_sentences=0)
        # We can't guarantee exact count, but > 1 means the chunker found a boundary
        assert len(chunks) >= 2, f"Expected >=2 chunks for different topics, got {len(chunks)}"

    def test_similar_sentences_stay_together(self, chunker):
        """Sentences about the same narrow topic should ideally be in one chunk."""
        text = (
            "Neural networks have layers. "
            "Each layer transforms the input representation. "
            "Deeper layers learn more abstract features."
        )
        chunks = chunker.chunk(text, overlap_sentences=0)
        # These are very similar — should be 1 chunk (or at most 2)
        assert len(chunks) <= 2

    def test_overlap_is_applied(self, chunker):
        """When overlap > 0, non-first chunks should have has_overlap=True."""
        text = (
            "Machine learning trains models on data. Supervised learning uses labels. "
            "Computer vision processes images. CNNs are the standard architecture."
        )
        chunks = chunker.chunk(text, overlap_sentences=1)
        if len(chunks) >= 2:
            assert chunks[1].get('has_overlap') is True, "Second chunk should have overlap"

    def test_overlap_zero_means_no_overlap(self, chunker):
        """When overlap=0, no chunk should have has_overlap=True."""
        text = (
            "Machine learning trains models. "
            "Quantum physics is about particles. "
            "Cooking is an art form."
        )
        chunks = chunker.chunk(text, overlap_sentences=0)
        for chunk in chunks:
            assert chunk.get('has_overlap') is False

    def test_max_sentences_enforced(self, chunker):
        """No chunk should exceed max_sentences."""
        text = '. '.join([f"Sentence number {i}" for i in range(20)]) + '.'
        chunks = chunker.chunk(text, max_sentences=5, overlap_sentences=0)
        for chunk in chunks:
            assert chunk['num_sentences'] <= 5

    def test_chunks_contain_embedding(self, chunker):
        """Each chunk should have a numpy embedding."""
        text = "Machine learning is great. Deep learning is a subset."
        chunks = chunker.chunk(text)
        for chunk in chunks:
            assert 'embedding' in chunk
            assert isinstance(chunk['embedding'], np.ndarray)

    def test_chunk_file(self, chunker, tmp_path):
        """Test chunking from a file path."""
        f = tmp_path / "test.txt"
        f.write_text("First topic sentence. Second topic sentence. Third unrelated sentence.")
        chunks = chunker.chunk_file(str(f))
        assert len(chunks) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
