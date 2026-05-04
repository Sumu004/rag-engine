"""
Hybrid Retrieval - Combines FAISS (dense) + BM25 (sparse) with Reciprocal Rank Fusion.

Key design decisions:
  - Incremental indexing: add_documents() appends to the existing corpus
    rather than replacing it, so ingesting a second file doesn't erase the
    first.
  - FAISS index is extended in-place via index.add(); BM25 must be rebuilt
    on the full corpus because BM25Okapi needs global IDF statistics.
  - save()/load() persist the FAISS index and a JSON sidecar containing
    the raw chunk texts and metadata.  The SentenceTransformer object is
    stored by *name* (not serialised), avoiding the JSON crash present in
    the previous version.
"""

import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

RRF_K = 60


class HybridRetriever:
    def __init__(self, embed_model: str = 'all-MiniLM-L6-v2'):
        self.embed_model_name: str = embed_model
        self.embed_model = SentenceTransformer(embed_model)
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()
        
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []
    
    def add_documents(self, chunks: List[Dict]):
        """Add chunked documents to the index *incrementally*.

        Unlike the previous implementation which replaced the entire
        corpus on every call, this version appends new chunks to the
        existing lists, encodes only the new embeddings, and extends
        the FAISS index in-place.

        BM25 must still be rebuilt from scratch because BM25Okapi
        computes global IDF statistics that change when the corpus
        grows.
        """
        if not chunks:
            return

        new_texts = [c['text'] for c in chunks]

        # Assign IDs that continue from the current corpus size.
        start_id = len(self.chunks)
        new_ids = [str(start_id + i) for i in range(len(chunks))]

        # Append to the running corpus.
        self.chunks.extend(new_texts)
        self.chunk_metadata.extend(chunks)
        self.doc_ids.extend(new_ids)

        # --- Dense index (FAISS) — encode only new chunks -----------------
        embeddings = self.embed_model.encode(new_texts, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        embeddings = embeddings / norms

        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(self.embed_dim)
        self.faiss_index.add(embeddings.astype('float32'))

        # --- Sparse index (BM25) — must rebuild for correct IDF -----------
        tokenized = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, k: int = 5, mode: str = 'hybrid') -> List[Dict]:
        """
        Search with different modes:
        - 'dense': FAISS only
        - 'sparse': BM25 only  
        - 'hybrid': RRF combination
        """
        if not self.chunks:
            return []
        
        if mode == 'dense':
            return self._search_dense(query, k)
        elif mode == 'sparse':
            return self._search_sparse(query, k)
        elif mode == 'hybrid':
            return self._search_hybrid(query, k)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _search_dense(self, query: str, k: int) -> List[Dict]:
        """FAISS semantic search."""
        query_vec = self.embed_model.encode([query], convert_to_numpy=True)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        scores, indices = self.faiss_index.search(query_vec.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                break
            results.append({
                'doc_id': self.doc_ids[idx],
                'text': self.chunks[idx],
                'score': float(score),
                'rank': len(results) + 1,
                'source': 'faiss'
            })
            if len(results) >= k:
                break
        
        return results
    
    def _search_sparse(self, query: str, k: int) -> List[Dict]:
        """BM25 keyword search."""
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'doc_id': self.doc_ids[idx],
                'text': self.chunks[idx],
                'score': float(scores[idx]),
                'rank': rank + 1,
                'source': 'bm25'
            })
        
        return results
    
    def _search_hybrid(self, query: str, k: int) -> List[Dict]:
        """Reciprocal Rank Fusion combination."""
        dense_results = self._search_dense(query, k * 2)
        sparse_results = self._search_sparse(query, k * 2)
        
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}
        
        for result in dense_results:
            doc_id = result['doc_id']
            rank = result['rank']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)
            doc_map[doc_id] = result
        
        for result in sparse_results:
            doc_id = result['doc_id']
            rank = result['rank']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = result
        
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:k]):
            result = doc_map[doc_id].copy()
            result['score'] = score
            result['rank'] = rank + 1
            result['source'] = 'hybrid'
            results.append(result)
        
        return results
    
    def classify_query(self, query: str) -> str:
        """Classify query type for routing."""
        query_lower = query.lower()
        
        factual_keywords = ['what is', 'how many', 'define', 'syntax', 'exact', 'list']
        semantic_keywords = ['explain', 'relationship', 'meaning', 'compare', 'describe']
        
        factual_count = sum(1 for kw in factual_keywords if kw in query_lower)
        semantic_count = sum(1 for kw in semantic_keywords if kw in query_lower)
        
        if factual_count > semantic_count and factual_count > 0:
            return 'sparse'
        elif semantic_count > factual_count:
            return 'dense'
        else:
            return 'hybrid'
    
    def index_stats(self) -> Dict:
        """Return index statistics."""
        return {
            'num_docs': len(self.chunks),
            'embedding_dim': self.embed_dim,
            'faiss_type': 'FlatIP',
            'bm25_type': 'Okapi',
            'faiss_ntotal': self.faiss_index.ntotal if self.faiss_index else 0,
        }
    
    def save(self, index_path: str):
        """Save index to disk.

        Persists the FAISS binary index and a JSON sidecar containing
        the raw chunk texts, metadata, and doc IDs.  The embedding
        model is stored by *name* so that load() can re-instantiate it
        without trying to serialise a PyTorch model to JSON.

        Numpy arrays in chunk_metadata (e.g. pre-computed embeddings)
        are stripped because they are not JSON-serialisable and can be
        recomputed from the text.
        """
        os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)

        if self.faiss_index:
            faiss.write_index(self.faiss_index, f"{index_path}.faiss")
        
        # Strip numpy arrays from metadata before serialising.
        clean_metadata = []
        for m in self.chunk_metadata:
            clean_metadata.append({
                k: v for k, v in m.items()
                if not isinstance(v, np.ndarray)
            })

        with open(f"{index_path}.json", 'w') as f:
            json.dump({
                'chunks': self.chunks,
                'metadata': clean_metadata,
                'doc_ids': self.doc_ids,
                'embed_model_name': self.embed_model_name,
            }, f)
    
    def load(self, index_path: str):
        """Load index from disk and rebuild BM25."""
        self.faiss_index = faiss.read_index(f"{index_path}.faiss")
        
        with open(f"{index_path}.json", 'r') as f:
            data = json.load(f)
            self.chunks = data['chunks']
            self.chunk_metadata = data['metadata']
            self.doc_ids = data['doc_ids']

        # Rebuild BM25 from the loaded corpus.
        if self.chunks:
            tokenized = [chunk.split() for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokenized)


def create_retriever(chunks: List[Dict]) -> HybridRetriever:
    """Convenience function to create and populate retriever."""
    retriever = HybridRetriever()
    retriever.add_documents(chunks)
    return retriever


if __name__ == '__main__':
    sample_chunks = [
        {'text': 'Python is a high-level programming language.', 'source': 'doc1'},
        {'text': 'Machine learning enables computers to learn from data.', 'source': 'doc2'},
        {'text': 'Deep learning uses neural networks with multiple layers.', 'source': 'doc3'},
        {'text': 'Natural language processing handles text data.', 'source': 'doc4'},
        {'text': 'FAISS is a library for efficient similarity search.', 'source': 'doc5'},
    ]
    
    retriever = HybridRetriever()
    retriever.add_documents(sample_chunks)
    
    print(f"Index: {retriever.index_stats()}")
    
    # Incremental add — should NOT erase the first batch.
    retriever.add_documents([
        {'text': 'Reinforcement learning trains agents via rewards.', 'source': 'doc6'},
    ])
    assert retriever.index_stats()['num_docs'] == 6, "Incremental add failed!"
    print(f"After incremental add: {retriever.index_stats()}")

    tests = [
        ("What is Python?", 'sparse'),
        ("Explain machine learning", 'dense'),
        ("neural networks", 'hybrid'),
    ]
    
    for query, expected_mode in tests:
        mode = retriever.classify_query(query)
        results = retriever.search(query, k=3, mode=mode)
        
        print(f"\nQuery: '{query}'")
        print(f"  Mode: {mode} (expected: {expected_mode})")
        print(f"  Results: {len(results)}")
        for r in results[:2]:
            print(f"    [{r['rank']}] {r['text'][:50]}... (score: {r['score']:.3f})")