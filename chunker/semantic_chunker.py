"""
Semantic Chunker - Splits documents by cosine similarity between sentence embeddings.
Unlike fixed-token chunking, this preserves semantic boundaries.
"""

import os
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

STRUCTURAL_PATTERNS = [
    (r'^#{1,6}\s+', 'heading'),
    (r'^```[\s\S]*?^```', 'code_block'),
    (r'^\d+\.\s+', 'numbered_list'),
    (r'^[-*]\s+', 'bulleted_list'),
    (r'\|.+\|.+\|', 'table'),
]

SIMILARITY_THRESHOLD = float(os.environ.get('CHUNK_SIMILARITY_THRESHOLD', '0.85'))


class SemanticChunker:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def is_structural(self, text: str) -> Optional[str]:
        """Check if text is a structural element that should not be split."""
        for pattern, element_type in STRUCTURAL_PATTERNS:
            if re.match(pattern, text, re.MULTILINE):
                return element_type
        return None
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic punctuation."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str, min_sentences: int = 1, max_sentences: int = 10) -> List[Dict]:
        """
        Chunk text by semantic similarity.
        
        Returns list of chunks with metadata:
        - text: chunk content
        - start_idx: character start position
        - end_idx: character end position
        - num_sentences: number of sentences in chunk
        """
        if not text or not text.strip():
            return []
        
        sentences = self.split_sentences(text)
        if len(sentences) == 0:
            return []
        
        if len(sentences) == 1:
            return [{
                'text': text,
                'start_idx': 0,
                'end_idx': len(text),
                'num_sentences': 1
            }]
        
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
        chunks = []
        current_chunk = [0]
        current_start = 0
        
        for i in range(1, len(sentences)):
            if i >= len(sentences):
                break
            
            sim = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
            structural_prev = self.is_structural(sentences[i-1])
            structural_curr = self.is_structural(sentences[i])
            
            if structural_prev != structural_curr:
                if len(current_chunk) >= min_sentences:
                    chunk_data = self._build_chunk(sentences, embeddings, current_chunk, current_start, i)
                    chunks.append(chunk_data)
                current_chunk = [i]
                current_start = i
            elif sim < SIMILARITY_THRESHOLD:
                if len(current_chunk) >= min_sentences:
                    chunk_data = self._build_chunk(sentences, embeddings, current_chunk, current_start, i)
                    chunks.append(chunk_data)
                current_chunk = [i]
                current_start = i
            elif len(current_chunk) >= max_sentences:
                chunk_data = self._build_chunk(sentences, embeddings, current_chunk, current_start, i)
                chunks.append(chunk_data)
                current_chunk = [i]
                current_start = i
            else:
                current_chunk.append(i)
        
        if current_chunk:
            chunk_data = self._build_chunk(sentences, embeddings, current_chunk, current_start, len(sentences))
            chunks.append(chunk_data)
        
        return chunks
    
    def _build_chunk(self, sentences: List[str], embeddings: np.ndarray, indices: List[int], 
                  start: int, end: int) -> Dict:
        """Build chunk from sentence indices."""
        chunk_text = ' '.join(sentences[i] for i in indices)
        chunk_embeddings = embeddings[indices]
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        
        start_pos = sum(len(sentences[i]) + 1 for i in range(start, min(indices)))
        end_pos = start_pos + len(chunk_text)
        
        return {
            'text': chunk_text,
            'start_idx': start_pos,
            'end_idx': end_pos,
            'num_sentences': len(indices),
            'embedding': avg_embedding
        }
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
    
    def chunk_file(self, file_path: str) -> List[Dict]:
        """Chunk a text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.chunk(text)
    
    def chunk_pdf(self, pdf_path: str) -> List[Dict]:
        """Chunk a PDF file."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf required for PDF chunking: pip install pypdf")
        
        reader = PdfReader(pdf_path)
        all_chunks = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                page_chunks = self.chunk(text)
                for chunk in page_chunks:
                    chunk['page'] = page_num + 1
                all_chunks.extend(page_chunks)
        
        return all_chunks


def chunk_text(text: str, min_sentences: int = 1, max_sentences: int = 10) -> List[Dict]:
    """ Convenience function for chunking text. """
    chunker = SemanticChunker()
    return chunker.chunk(text, min_sentences, max_sentences)


if __name__ == '__main__':
    sample_text = """
    Machine learning is a subset of artificial intelligence. It focuses on training models to make predictions.
    Deep learning uses neural networks with multiple layers. These networks can learn hierarchical representations.
    Natural language processing applies these techniques to text data. Sentiment analysis is a common NLP task.
    Computer vision leverages CNNs for image classification. Object detection identifies multiple objects in images.
    """
    
    chunker = SemanticChunker()
    chunks = chunker.chunk(sample_text)
    
    print(f"Input: {len(sample_text)} chars, {len(sample_text.split())} words")
    print(f"Output: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}] ({chunk['num_sentences']} sentences)")
        print(f"  {chunk['text'][:100]}...")