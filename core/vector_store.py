"""
FAISS Vector Store Module.
Manages vector indexing, L2 normalization for Cosine Similarity,
and Top-K nearest-neighbor retrieval for the RAG pipeline.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import faiss


class FAISSVectorStore:
    """
    In-memory vector database powered by FAISS (Facebook AI Similarity Search).
    Uses IndexFlatIP with L2-normalized embeddings to compute exact Cosine Similarity.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Builds a new FAISS index from document chunks and their dense embeddings.
        
        Args:
            chunks: List of chunk metadata dicts (from pdf_processor.chunk_text)
            embeddings: 2D numpy array of shape (N, dimension), dtype float32
        """
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            self.clear()
            return

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(chunks)} chunks provided, but embeddings has {embeddings.shape[0]} rows.")

        dim = embeddings.shape[1]
        self.dimension = dim

        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype="float32")
        
        # Normalize vectors for Cosine Similarity via Inner Product
        faiss.normalize_L2(vectors)

        # Create IndexFlatIP (Inner Product = Cosine Similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.chunks = list(chunks)

    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 4, 
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Performs Top-K Cosine Similarity search for a given query vector.
        
        Args:
            query_vector: 1D or 2D numpy array representing the query embedding.
            top_k: Number of most relevant chunks to retrieve.
            min_similarity: Minimum cosine similarity score threshold (0.0 to 1.0).
            
        Returns:
            List of retrieved chunks with similarity scores:
            [{
                "chunk_id": int,
                "page_number": int,
                "text": str,
                "char_count": int,
                "similarity_score": float
            }, ...]
        """
        if self.index is None or self.index.ntotal == 0 or len(self.chunks) == 0:
            return []

        # Format query vector to (1, dimension)
        q_vec = np.ascontiguousarray(query_vector, dtype="float32")
        if q_vec.ndim == 1:
            q_vec = np.expand_dims(q_vec, axis=0)

        # Normalize query vector
        faiss.normalize_L2(q_vec)

        # Constrain top_k to total available chunks
        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return []

        distances, indices = self.index.search(q_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            score = float(dist)
            if score >= min_similarity:
                chunk_data = dict(self.chunks[idx])
                chunk_data["similarity_score"] = round(score, 4)
                results.append(chunk_data)

        return results

    def is_empty(self) -> bool:
        """Returns True if the index contains no vectors."""
        return self.index is None or self.index.ntotal == 0

    def get_stats(self) -> Dict[str, Any]:
        """Returns current index statistics."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": "faiss.IndexFlatIP (Cosine Similarity)"
        }

    def clear(self) -> None:
        """Resets the vector index and chunks list."""
        self.index = None
        self.chunks = []
