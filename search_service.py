"""
Search Service for PubMed Semantic Search
Contains all search-related functionality including embedding generation and hybrid search.
"""

import asyncio
import os
import numpy as np
import streamlit as st
from typing import List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv

# Import required modules
from embeddings import (
    TextEmbedder as EmbeddingService, 
    get_available_models,
    PubMedBERTEmbedder,
    BioBERTEmbedder,
    GeminiEmbedder
)
from improved_vector_store import ImprovedVectorStore
from rag_pipeline import ultra_fast_chunking
from reranker_flashrank import flashrank_rerank

# Load environment variables
load_dotenv()

# Model configuration is now handled by the unified embedding service

def get_embedding_service(model_name: str) -> EmbeddingService:
    """Get cached embedding service instance with the specified model."""
    # Get from cache or create new
    if not hasattr(st.session_state, 'embedding_services'):
        st.session_state.embedding_services = {}
    
    if model_name not in st.session_state.embedding_services:
        # Use specialized embedders for specific models
        if 'pubmedbert' in model_name.lower() or 'biomednlp' in model_name.lower():
            st.session_state.embedding_services[model_name] = PubMedBERTEmbedder()
        elif 'biobert' in model_name.lower():
            st.session_state.embedding_services[model_name] = BioBERTEmbedder()
        elif 'gemini' in model_name.lower():
            st.session_state.embedding_services[model_name] = GeminiEmbedder()
        else:
            # Fall back to generic TextEmbedder for other models
            st.session_state.embedding_services[model_name] = EmbeddingService(
                model_name=model_name
            )
    
    return st.session_state.embedding_services[model_name]

@st.cache_data(show_spinner=False)
def cached_embeddings_chunked(
    key: str,
    texts: List[str],
    model_name: str,
    backend: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    batch_size: int = 16,
) -> np.ndarray:
    """Generate cached embeddings with chunking using the unified embedding service"""
    # Get the embedding service
    embedder = get_embedding_service(model_name)
    
    # Prepare chunked texts per document
    chunked: List[List[str]] = []
    chunked_indices: List[int] = []
    
    for i, text in enumerate(texts):
        chunks = ultra_fast_chunking(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked.extend(chunks)
        chunked_indices.extend([i] * len(chunks))
    
    # Generate embeddings in batches
    all_embeddings = []
    for i in range(0, len(chunked), batch_size):
        batch = chunked[i:i + batch_size]
        embeddings = embedder.encode(batch, batch_size=len(batch), normalize=True)
        all_embeddings.append(embeddings)
    
    if not all_embeddings:
        return np.array([])
        
    # Combine all embeddings and group by original document
    all_embeddings = np.vstack(all_embeddings)
    doc_embeddings = []
    
    for i in range(len(texts)):
        doc_chunk_indices = [idx for idx, doc_idx in enumerate(chunked_indices) if doc_idx == i]
        if not doc_chunk_indices:
            # If no chunks were generated for this document, use zeros
            doc_embeddings.append(np.zeros(embedder.get_embedding_dimensions()))
        else:
            # Average the embeddings of all chunks for this document
            doc_embedding = np.mean(all_embeddings[doc_chunk_indices], axis=0)
            # L2 normalize the final document embedding
            doc_embedding = doc_embedding / np.linalg.norm(doc_embedding, axis=-1, keepdims=True)
            doc_embeddings.append(doc_embedding)
    
    return np.array(doc_embeddings)

def generate_query_embedding(query: str, model_name: str, backend: str, doc_embeddings_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Generate query embedding using the unified embedding service"""
    if not query:
        return None
        
    try:
        # Get the embedding service
        embedder = get_embedding_service(model_name)
        
        # Get query embedding using the unified interface
        q_vec = embedder.encode([query], batch_size=1, normalize=True)[0]
        
        # Ensure query embedding matches document embedding dimensions
        if len(q_vec) != doc_embeddings_shape[1]:
            st.warning(f"Query embedding dimension ({len(q_vec)}) does not match document embedding dimension ({doc_embeddings_shape[1]})")
            return None
            
        return q_vec.reshape(1, -1)  # Return as 2D array for compatibility
        
    except Exception as e:
        st.error(f"Error generating query embedding: {str(e)}")
        return None

def build_vector_store(texts: List[str], doc_embeddings: np.ndarray, metadata: List[Dict]) -> ImprovedVectorStore:
    """Build and configure the vector store"""
    vector_store = ImprovedVectorStore()
    # Fixed, simple weights
    vector_store.semantic_weight = 0.8
    vector_store.keyword_weight = 0.2
    effective_index_type = "flat" if len(texts) < 300 else "ivf"
    vector_store.build_hybrid_index(texts, doc_embeddings, metadata, effective_index_type)
    return vector_store

def perform_hybrid_search(
    vector_store: ImprovedVectorStore,
    query: str,
    query_embedding: np.ndarray,
    top_k: int,
    use_reranking: bool
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Perform hybrid search using the vector store"""
    scores, indices, result_metadata = vector_store.hybrid_search(
        query, query_embedding, top_k=top_k, use_reranking=use_reranking
    )
    return scores, indices, result_metadata

def apply_flashrank_reranking(
    query: str,
    articles: List,
    keep_indices: List[int],
    scores: np.ndarray,
    indices: np.ndarray,
    result_metadata: List[Any]
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Apply FlashRank reranking if enabled"""
    try:
        return flashrank_rerank(
            query=query,
            articles=articles,
            keep_indices=keep_indices,
            scores=scores,
            indices=indices,
            result_metadata=result_metadata,
        )
    except Exception as e:
        st.warning(f"FlashRank rerank failed: {e}")
        return scores, indices, result_metadata

def sort_search_results(
    scores: np.ndarray,
    indices: np.ndarray,
    result_metadata: List[Any],
    articles: List,
    keep_indices: List[int],
    sort_by: str,
    sort_order: str
) -> List[Dict]:
    """Sort search results based on user selection"""
    sorted_results = []
    for score, idx, meta in zip(scores, indices, result_metadata):
        if idx < 0 or idx >= len(keep_indices):
            continue
        global_idx = keep_indices[idx]
        art = articles[global_idx]
        
        # Prepare sorting data
        sort_data = {
            "score": score,
            "idx": idx,
            "meta": meta,
            "art": art,
            "relevance_score": float(score),
            "publication_date": art.year or "0",
            "journal_name": art.journal or "",
            "title_alphabetical": art.title or "",
            "semantic_score": getattr(meta, 'semantic_score', 0),
            "keyword_score": getattr(meta, 'keyword_score', 0)
        }
        sorted_results.append(sort_data)
    
    # Sort the results
    reverse_sort = (sort_order == "desc")
    if sort_by == "relevance_score":
        sorted_results.sort(key=lambda x: x["relevance_score"], reverse=reverse_sort)
    elif sort_by == "publication_date":
        sorted_results.sort(key=lambda x: int(x["publication_date"]) if x["publication_date"].isdigit() else 0, reverse=reverse_sort)
    elif sort_by == "journal_name":
        sorted_results.sort(key=lambda x: x["journal_name"].lower(), reverse=reverse_sort)
    elif sort_by == "title_alphabetical":
        sorted_results.sort(key=lambda x: x["title_alphabetical"].lower(), reverse=reverse_sort)
    elif sort_by == "semantic_score":
        sorted_results.sort(key=lambda x: x["semantic_score"], reverse=reverse_sort)
    elif sort_by == "keyword_score":
        sorted_results.sort(key=lambda x: x["keyword_score"], reverse=reverse_sort)

    return sorted_results
