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
from embeddings import TextEmbedder
from improved_vector_store import ImprovedVectorStore
from rag_pipeline import ultra_fast_chunking
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from reranker_flashrank import flashrank_rerank

# Load environment variables
load_dotenv()

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str, backend: str) -> TextEmbedder:
    """Get cached embedder instance"""
    use_st = model_name.startswith("sentence-transformers/") or backend.lower().startswith("sentence")
    return TextEmbedder(model_name=model_name, use_sentence_transformers=use_st)

@st.cache_data(show_spinner=False)
def cached_embeddings_chunked(
    key: str,
    texts: List[str],
    model_name: str,
    backend: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> np.ndarray:
    """Generate cached embeddings with chunking"""
    # Prepare chunked texts per document
    chunked: List[List[str]] = []
    for t in texts:
        t_str = t if isinstance(t, str) else str(t)
        chunks = ultra_fast_chunking(t_str, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked.append(chunks if len(chunks) > 0 else [t_str])

    # Flatten for embedding
    flat_texts: List[str] = [c for doc in chunked for c in doc]
    if len(flat_texts) == 0:
        return np.zeros((0, 768), dtype=np.float32)

    if model_name == "gemini":
        try:
            # Ensure an event loop exists for Gemini client
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            flat_vectors: List[List[float]] = emb.embed_documents(flat_texts)
            flat_arr = np.array(flat_vectors, dtype=np.float32)
        except Exception as e:
            # Try direct Google Generative AI client as fallback
            try:
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                flat_vectors = []
                for t in flat_texts:
                    res = genai.embed_content(model="models/embedding-001", content=t)
                    vec = res.get("embedding") or res.get("data", [{}])[0].get("embedding")
                    if vec is None:
                        raise RuntimeError("No embedding returned from Gemini API")
                    flat_vectors.append(vec)
                flat_arr = np.array(flat_vectors, dtype=np.float32)
            except Exception as e2:
                print(f"Gemini embeddings failed: {e}; direct API fallback failed: {e2}. Falling back to PubMedBERT")
                embedder = TextEmbedder(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", use_sentence_transformers=False)
                flat_arr = embedder.encode(flat_texts, batch_size=16, normalize=True)
    elif model_name.startswith("sentence-transformers/"):
        # Use Sentence Transformers directly, with local caching
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, cache_folder="models")
            embeddings = model.encode(flat_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            flat_arr = np.array(embeddings, dtype=np.float32)
        except Exception as e:
            print(f"Sentence Transformers failed: {e}, falling back to PubMedBERT")
            embedder = TextEmbedder(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", use_sentence_transformers=False)
            flat_arr = embedder.encode(flat_texts, batch_size=16, normalize=True)
    else:
        embedder = get_embedder(model_name, backend)
        flat_arr = embedder.encode(flat_texts, batch_size=16, normalize=True)

    # Average pool chunks per original document
    emb_dim = flat_arr.shape[1]
    doc_embeddings = np.zeros((len(chunked), emb_dim), dtype=np.float32)
    cursor = 0
    for i, ch in enumerate(chunked):
        num = len(ch)
        doc_vecs = flat_arr[cursor:cursor+num]
        cursor += num
        if doc_vecs.shape[0] == 1:
            avg = doc_vecs[0]
        else:
            avg = doc_vecs.mean(axis=0)
        # Normalize
        norm = np.linalg.norm(avg) + 1e-12
        doc_embeddings[i] = (avg / norm).astype(np.float32)

    return doc_embeddings

def generate_query_embedding(query: str, model_name: str, backend: str, doc_embeddings_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Generate query embedding using the same model as document embeddings"""
    def ensure_query_shape(q_emb, dim):
        q_emb = np.array(q_emb, dtype=np.float32)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        if q_emb.shape[1] != dim:
            st.error(f"Query embedding dimension {q_emb.shape[1]} does not match index dimension {dim}.")
            return None
        return q_emb

    if model_name == "gemini":
        try:
            # Ensure an event loop exists for Gemini client
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            q_vec = emb.embed_query(query)
            query_embedding = ensure_query_shape(q_vec, doc_embeddings_shape[1])
        except Exception as e:
            # Try direct Google Generative AI client as fallback
            try:
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                res = genai.embed_content(model="models/embedding-001", content=query)
                q_vec = np.array(res.get("embedding") or res.get("data", [{}])[0].get("embedding"), dtype=np.float32)
                query_embedding = ensure_query_shape(q_vec, doc_embeddings_shape[1])
            except Exception as e2:
                print(f"Gemini query embedding failed: {e}; direct API fallback failed: {e2}. Falling back to PubMedBERT")
                embedder = TextEmbedder(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", use_sentence_transformers=False)
                q_vec = embedder.encode([query], batch_size=1, normalize=True)
                query_embedding = ensure_query_shape(q_vec, doc_embeddings_shape[1])
    elif model_name.startswith("sentence-transformers/"):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, cache_folder="models")
            q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            query_embedding = ensure_query_shape(q_vec, doc_embeddings_shape[1])
        except Exception as e:
            print(f"Sentence Transformers query embedding failed: {e}, falling back to PubMedBERT")
            embedder = TextEmbedder(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", use_sentence_transformers=False)
            q_vec = embedder.encode([query], batch_size=1, normalize=True)
            query_embedding = ensure_query_shape(q_vec, doc_embeddings_shape[1])
    else:
        embedder = get_embedder(model_name, backend)
        q_vec = embedder.encode([query], batch_size=1, normalize=True)
        query_embedding = ensure_query_shape(q_vec, doc_embeddings_shape[1])

    return query_embedding

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
        # Show top 5 titles before rerank
        st.write("🔹 Before FlashRank (top 5):", [
            m.get("title") if isinstance(m, dict) else getattr(m, "title", "") 
            for m in result_metadata[:5]
        ])

        scores, indices, result_metadata = flashrank_rerank(
            query=query,
            articles=articles,
            keep_indices=keep_indices,
            scores=scores,
            indices=indices,
            result_metadata=result_metadata,
        )

        st.write("🔹 After FlashRank (top 5):", [
            m.get("title") if isinstance(m, dict) else getattr(m, "title", "") 
            for m in result_metadata[:5]
        ])

        return scores, indices, result_metadata
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
