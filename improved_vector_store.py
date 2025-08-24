from __future__ import annotations

import os
import pickle
import json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import re
import time

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ImprovedVectorStore:
    """
    Enhanced vector store with persistence, hybrid search, and better indexing
    """
    
    def __init__(self, storage_dir: str = "vector_cache"):
        # Use absolute path under the module directory to avoid CWD issues
        base_dir = Path(__file__).parent
        storage_path = Path(storage_dir)
        if not storage_path.is_absolute():
            storage_path = base_dir / storage_path
        self.storage_dir = storage_path
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # FAISS index for semantic search
        self.faiss_index: Optional[faiss.Index] = None
        
        # TF-IDF for keyword search
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        
        # Metadata storage
        self.article_metadata: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Search weights
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3
        
    def build_hybrid_index(
        self, 
        texts: List[str], 
        embeddings: np.ndarray,
        metadata: List[Dict],
        index_type: str = "ivf"
    ) -> None:
        """
        Build both FAISS semantic index and TF-IDF keyword index
        
        Args:
            texts: List of text documents (abstracts)
            embeddings: Pre-computed embeddings
            metadata: Article metadata
            index_type: FAISS index type ("flat", "ivf", "hnsw")
        """
        self.embeddings = embeddings.astype(np.float32)
        self.article_metadata = metadata
        
        # Build FAISS index based on type
        if index_type == "flat":
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        elif index_type == "ivf":
            # IVF index for better speed with clustering
            nlist = min(100, max(1, embeddings.shape[0] // 10))  # Number of clusters
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
            self.faiss_index.train(embeddings)
        elif index_type == "hnsw":
            # HNSW for approximate nearest neighbor with good accuracy
            self.faiss_index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)  # 32 neighbors
            self.faiss_index.hnsw.efConstruction = 200
            self.faiss_index.hnsw.efSearch = 100
        
        self.faiss_index.add(embeddings)
        
        # Build TF-IDF index for keyword search
        # Adjust min_df based on dataset size to avoid "no terms remain" error
        min_df = max(1, min(2, len(texts) // 2))  # At least 1, but no more than 2
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.95
        )
        if len(texts) < 5:  
    # very small dataset
            self.tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
        else:
    # normal dataset
            self.tfidf_vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.9)

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        print(f"Built hybrid index: {embeddings.shape[0]} documents, {embeddings.shape[1]} dimensions")
        
    def hybrid_search(
        self, 
        query: str, 
        query_embedding: np.ndarray,
        top_k: int = 10,
        use_reranking: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Perform hybrid semantic + keyword search with optional reranking
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            top_k: Number of results to return
            use_reranking: Whether to use reranking for better results
            
        Returns:
            scores, indices, metadata
        """
        # Semantic search
        semantic_scores, semantic_indices = self._semantic_search(query_embedding, top_k * 2)
        
        # Keyword search
        keyword_scores, keyword_indices = self._keyword_search(query, top_k * 2)
        
        # Combine results
        combined_results = self._combine_search_results(
            semantic_scores, semantic_indices, 
            keyword_scores, keyword_indices, 
            top_k
        )
        
        if use_reranking and len(combined_results) > 1:
            combined_results = self._rerank_results(query, combined_results)
        
        # Extract final results
        final_scores = np.array([r['final_score'] for r in combined_results[:top_k]])
        final_indices = np.array([r['index'] for r in combined_results[:top_k]])
        final_metadata = [self.article_metadata[idx] for idx in final_indices]
        
        return final_scores, final_indices, final_metadata
    
    def _semantic_search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform semantic search using FAISS"""
        if self.faiss_index is None:
            return np.array([]), np.array([])
        
        # Normalize query embedding
        query_norm = query_embedding.astype(np.float32)
        if query_norm.shape[0] == 1:
            query_norm = query_norm.reshape(1, -1)
        
        scores, indices = self.faiss_index.search(query_norm, top_k)
        return scores[0], indices[0]
    
    def _keyword_search(self, query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform keyword search using TF-IDF"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return np.array([]), np.array([])
        
        # Vectorize query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_scores, top_indices
    
    def _combine_search_results(
        self, 
        semantic_scores: np.ndarray, 
        semantic_indices: np.ndarray,
        keyword_scores: np.ndarray, 
        keyword_indices: np.ndarray,
        top_k: int
    ) -> List[Dict]:
        """Combine semantic and keyword search results"""
        combined = {}
        
        # Add semantic results
        for score, idx in zip(semantic_scores, semantic_indices):
            if idx >= 0 and idx < len(self.article_metadata):
                combined[idx] = {
                    'index': idx,
                    'semantic_score': float(score),
                    'keyword_score': 0.0,
                    'final_score': float(score) * self.semantic_weight
                }
        
        # Add keyword results
        for score, idx in zip(keyword_scores, keyword_indices):
            if idx >= 0 and idx < len(self.article_metadata):
                if idx in combined:
                    combined[idx]['keyword_score'] = float(score)
                    combined[idx]['final_score'] += float(score) * self.keyword_weight
                else:
                    combined[idx] = {
                        'index': idx,
                        'semantic_score': 0.0,
                        'keyword_score': float(score),
                        'final_score': float(score) * self.keyword_weight
                    }
        
        # Sort by final score
        results = list(combined.values())
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results[:top_k * 2]  # Return more for reranking
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using additional signals"""
        for result in results:
            idx = result['index']
            metadata = self.article_metadata[idx]
            
            # Boost recent papers
            if 'year' in metadata and metadata['year']:
                try:
                    year = int(metadata['year'])
                    current_year = 2024
                    if year >= current_year - 2:
                        result['final_score'] *= 1.1  # 10% boost for recent papers
                except:
                    pass
            
            # Boost papers with longer abstracts (more content)
            if 'abstract' in metadata:
                abstract_length = len(metadata['abstract'])
                if abstract_length > 500:
                    result['final_score'] *= 1.05  # 5% boost for detailed abstracts
            
            # Boost papers from high-impact journals (simple heuristic)
            if 'journal' in metadata:
                journal = metadata['journal'].lower()
                high_impact = ['nature', 'science', 'cell', 'lancet', 'nejm', 'jama']
                if any(hj in journal for hj in high_impact):
                    result['final_score'] *= 1.15  # 15% boost for high-impact journals
        
        # Resort by updated scores
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
    
    def save_index(self, filename: str) -> None:
        """Save the index and metadata to disk"""
        # Sanitize filename for Windows compatibility
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename or "index")
        save_path = self.storage_dir / safe_name
        
        # Save FAISS index
        if self.faiss_index is not None:
            faiss_path = (save_path.with_suffix('.faiss')).resolve()
            tmp_faiss = faiss_path.with_suffix('.faiss.tmp')
            # Write to temp file then atomically replace
            faiss.write_index(self.faiss_index, str(tmp_faiss))
            if faiss_path.exists():
                faiss_path.unlink()
            tmp_faiss.rename(faiss_path)
        
        # Save metadata and other components
        metadata = {
            'article_metadata': self.article_metadata,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'embeddings': self.embeddings,
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight
        }
        
        pkl_path = (save_path.with_suffix('.pkl')).resolve()
        tmp_pkl = pkl_path.with_suffix('.pkl.tmp')
        with open(tmp_pkl, 'wb') as f:
            pickle.dump(metadata, f)
        if pkl_path.exists():
            pkl_path.unlink()
        tmp_pkl.rename(pkl_path)
        
        print(f"Index saved to {save_path}")
    
    def load_index(self, filename: str) -> bool:
        """Load the index and metadata from disk"""
        load_path = self.storage_dir / filename
        
        try:
            # Load FAISS index
            faiss_path = load_path.with_suffix('.faiss')
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
            
            # Load metadata
            pkl_path = load_path.with_suffix('.pkl')
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.article_metadata = metadata['article_metadata']
                self.tfidf_vectorizer = metadata['tfidf_vectorizer']
                self.tfidf_matrix = metadata['tfidf_matrix']
                self.embeddings = metadata['embeddings']
                self.semantic_weight = metadata.get('semantic_weight', 0.7)
                self.keyword_weight = metadata.get('keyword_weight', 0.3)
                
                print(f"Index loaded: {len(self.article_metadata)} documents")
                return True
                
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False
        
        return False
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index"""
        stats = {
            'total_documents': len(self.article_metadata),
            'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'faiss_index_type': type(self.faiss_index).__name__ if self.faiss_index else None,
            'has_tfidf': self.tfidf_vectorizer is not None,
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight
        }
        return stats

    # Convenience API to match "top_n" naming
    def retrieve_top_n(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_n: int = 10,
        use_reranking: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Retrieve top-N results using hybrid search."""
        return self.hybrid_search(query, query_embedding, top_k=top_n, use_reranking=use_reranking)


# Backward compatibility functions
def build_faiss_ip_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Legacy function for backward compatibility"""
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_index(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy function for backward compatibility"""
    if query_embeddings.dtype != np.float32:
        query_embeddings = query_embeddings.astype(np.float32, copy=False)
    scores, indices = index.search(query_embeddings, top_k)
    return scores, indices
