from __future__ import annotations

from reranker_flashrank import flashrank_rerank
import hashlib
from typing import Dict, List, Optional
import streamlit as st
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

from embeddings import TextEmbedder
from pubmed_fetcher import PubMedArticle, fetch_pubmed_articles
from improved_vector_store import ImprovedVectorStore
from rag_pipeline import make_text_chunks
from query_processing import expand_query
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai  
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

st.set_page_config(page_title="PubMed Semantic Search (Improved)", layout="wide")

# Enhanced styles for better UI
st.markdown(
    """
    <style>
    .result-card {
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .result-title { 
        font-size: 1.1rem; 
        font-weight: 700; 
        margin-bottom: 8px; 
        color: #1f2937;
    }
    .result-meta { 
        color: #6b7280; 
        font-size: 0.9rem; 
        margin-bottom: 10px; 
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .result-score { 
        color: #2563eb; 
        font-weight: 700; 
        font-size: 1.1rem;
    }
    .result-abstract { 
        color: #374151; 
        font-size: 0.95rem; 
        line-height: 1.6;
        margin-bottom: 12px;
    }
    .score-breakdown {
        background: #f3f4f6;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #4b5563;
    }
    .metric-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .semantic-badge { background: #dbeafe; color: #1e40af; }
    .keyword-badge { background: #fef3c7; color: #92400e; }
    .rerank-badge { background: #dcfce7; color: #166534; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_MODELS: Dict[str, str] = {
    "Gemini (models/embedding-001)": "gemini",
    "Sentence Transformers (all-MiniLM-L6-v2)": "sentence-transformers/all-MiniLM-L6-v2",
    "PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "BioBERT (dmis-lab/biobert-base-cased-v1.1)": "dmis-lab/biobert-base-cased-v1.1",
}

SYNONYMS: Dict[str, List[str]] = {
    "heart": ["cardiac", "cardio", "myocardial"],
    "attack": ["infarction", "acute coronary syndrome"],
    "stroke": ["cerebrovascular accident", "brain attack"],
    "diabetes": ["hyperglycemia", "type 2 diabetes", "t2d", "diabetic"],
    "cancer": ["malignancy", "neoplasm", "tumor", "carcinoma"],
    "hypertension": ["high blood pressure", "htn"],
    "depression": ["major depressive disorder", "mdd", "clinical depression"],
    "alzheimer": ["dementia", "cognitive decline", "memory loss"],
}

def expand_query_simple(query: str, synonyms: Dict[str, List[str]]) -> str:
    """Enhanced query expansion with better synonym handling"""
    tokens = query.lower().split()
    expanded_terms: List[str] = []
    
    for t in tokens:
        key = t.strip('.,;:?!')
        syns = synonyms.get(key, [])
        if syns:
            # Use OR operator for synonyms
            parts = [t] + syns
            expanded_terms.append("(" + " OR ".join(parts) + ")")
        else:
            expanded_terms.append(t)
    
    return " ".join(expanded_terms)

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str, backend: str) -> TextEmbedder:
    # For Sentence Transformers models, use sentence_transformers backend
    # For other models, use transformers backend
    use_st = model_name.startswith("sentence-transformers/") or backend.lower().startswith("sentence")
    return TextEmbedder(model_name=model_name, use_sentence_transformers=use_st)

@st.cache_data(show_spinner=False)
def cached_fetch_pubmed(query: str, retmax: int, email: Optional[str], api_key: Optional[str]) -> List[PubMedArticle]:
    return fetch_pubmed_articles(query=query, retmax=retmax, email=email, api_key=api_key)

@st.cache_data(show_spinner=False)
def cached_embeddings_chunked(
    key: str,
    texts: List[str],
    model_name: str,
    backend: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> np.ndarray:
    # Prepare chunked texts per document
    chunked: List[List[str]] = []
    for t in texts:
        t_str = t if isinstance(t, str) else str(t)
        chunks = make_text_chunks(t_str, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
            from embeddings import get_embedding_model
            model = get_embedding_model(model_name, local_base_dir="models")
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

@st.cache_data(show_spinner=False)
def _hash_key(*parts: str) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(p.encode("utf-8"))
    return m.hexdigest()

def main() -> None:
    st.title("🔬 PubMed Semantic Search (Enhanced)")
    st.caption("Advanced biomedical literature search with hybrid semantic + keyword search, persistence, and intelligent reranking.")

    with st.sidebar:
        st.subheader("⚙️ Settings")
        email = st.text_input("NCBI Entrez Email (recommended)", placeholder="your.email@domain.com")
        
        st.divider()
        
        st.subheader("🧠 AI Model Settings")
        model_label = st.selectbox("Embedding model", list(DEFAULT_MODELS.keys()))
        model_name = DEFAULT_MODELS[model_label]
        backend = "Sentence-Transformers" if model_name.startswith("sentence-transformers/") else "Transformers"
        
        # Show model info
        if model_name == "gemini":
            st.info("🌐 **Gemini Embeddings** - Gemini embedding 001 is used.")
        elif model_name.startswith("sentence-transformers/"):
            st.info("🔤 **Sentence Transformers** - Fast, lightweight embeddings optimized for semantic similarity.")
        else:
            st.info("🧬 **Biomedical BERT** - Domain-specific models trained on biomedical literature.")
        
        st.divider()
        
        st.subheader("📊 Search Configuration")
        retmax = st.slider("Articles to fetch", 20, 500, 100, step=20)
        top_k = st.slider("Top-N results", 5, 50, 15)
        
        st.divider()
        
        st.subheader("🔍 Search Enhancement")
        expand = st.checkbox("Expand query with synonyms", value=True, help="Use medical synonyms for better coverage")
        use_reranking = st.checkbox("Enable intelligent reranking", value=True, help="Boost recent, high-impact papers")
        use_flashrank = st.sidebar.checkbox("Use Langchain's FlashRank reranker", value=False)

        
        st.divider()
        
        st.subheader("💾 Persistence")
        save_index = st.checkbox("Save index for reuse", value=True, help="Save embeddings to avoid recomputation")
        index_name = st.text_input("Index name", value="pubmed_index", help="Name for saved index")
        
        clear_cache = st.button("🗑️ Clear all cache", type="primary")
        if clear_cache:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared successfully!")

    # Main search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("🔍 Enter your medical query", 
                             placeholder="e.g., heart attack symptoms, diabetes complications, cancer immunotherapy")
    with col2:
        do_search = st.button("🚀 Search", type="primary", use_container_width=True)

    if do_search and query.strip():
        # Query expansion
        if expand:
            try:
                run_query, synonyms_map, tokens = expand_query(query, email=email or "")
                st.info(f"🔍 **Expanded query:** {run_query}")
                with st.expander("🔍 Query Expansion Details", expanded=False):
                    st.write("Tokens:", tokens)
                    st.write("Top synonyms per token (truncated):")
                    preview = {k: v[:5] for k, v in synonyms_map.items()}
                    st.json(preview)
            except Exception as e:
                st.warning(f"⚠️ Query expansion failed: {e}. Using original query.")
                run_query = query
        else:
            run_query = query

        # Ensure email is set
        email_effective = (email or "").strip() or "pubmed-semantic@example.com"

        # Fetch articles
        with st.spinner("📚 Fetching PubMed articles..."):
            try:
                articles = cached_fetch_pubmed(run_query, retmax, email_effective, None)
            except Exception as e:
                st.error(f"❌ PubMed request failed: {e}")
                return

        if not articles:
            st.warning("⚠️ No results from PubMed. Try a different query.")
            return

        st.success(f"✅ Retrieved {len(articles)} articles from PubMed")

        # Prepare texts and metadata
        texts: List[str] = []
        keep_indices: List[int] = []
        metadata: List[Dict] = []
        
        for idx, art in enumerate(articles):
            if isinstance(art.abstract, str) and art.abstract.strip():
                texts.append(art.abstract.strip())
                keep_indices.append(idx)
                metadata.append({
                    'title': art.title,
                    'abstract': art.abstract,
                    'pmid': art.pmid,
                    'url': art.url,
                    'journal': art.journal,
                    'year': art.year,
                    'authors': art.authors
                })

        if not texts:
            st.warning("⚠️ Fetched articles have no abstracts to embed.")
            return

        # Generate embeddings (chunked + averaged)
        emb_key = _hash_key("embeddings_chunked", run_query, model_name, backend, str(retmax))
        with st.spinner("🧠 Generating embeddings (chunked)..."):
            doc_embeddings = cached_embeddings_chunked(emb_key, texts, model_name, backend)

        # Build enhanced vector store
        with st.spinner("🔧 Building hybrid search index..."):
            vector_store = ImprovedVectorStore()
            # Fixed, simple weights
            vector_store.semantic_weight = 0.8
            vector_store.keyword_weight = 0.2
            effective_index_type = "flat" if len(texts) < 300 else "ivf"
            vector_store.build_hybrid_index(texts, doc_embeddings, metadata, effective_index_type)

        # Save index if requested
        if save_index:
            try:
                vector_store.save_index(index_name)
                st.success(f"💾 Index saved as '{index_name}'")
            except Exception as e:
                st.warning(f"⚠️ Failed to save index: {e}")

        # Perform hybrid search
        with st.spinner("🔍 Performing hybrid search..."):
            # Always use the same model for query embedding as for document embedding
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
                    query_embedding = ensure_query_shape(q_vec, doc_embeddings.shape[1])
                except Exception as e:
                    # Try direct Google Generative AI client as fallback
                    try:
                        api_key = os.environ.get("GOOGLE_API_KEY")
                        if api_key:
                            genai.configure(api_key=api_key)
                        res = genai.embed_content(model="models/embedding-001", content=query)
                        q_vec = np.array(res.get("embedding") or res.get("data", [{}])[0].get("embedding"), dtype=np.float32)
                        query_embedding = ensure_query_shape(q_vec, doc_embeddings.shape[1])
                    except Exception as e2:
                        print(f"Gemini query embedding failed: {e}; direct API fallback failed: {e2}. Falling back to PubMedBERT")
                        embedder = TextEmbedder(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", use_sentence_transformers=False)
                        q_vec = embedder.encode([query], batch_size=1, normalize=True)
                        query_embedding = ensure_query_shape(q_vec, doc_embeddings.shape[1])
            elif model_name.startswith("sentence-transformers/"):
                try:
                    from embeddings import get_embedding_model
                    model = get_embedding_model(model_name, local_base_dir="models")
                    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                    query_embedding = ensure_query_shape(q_vec, doc_embeddings.shape[1])
                except Exception as e:
                    print(f"Sentence Transformers query embedding failed: {e}, falling back to PubMedBERT")
                    embedder = TextEmbedder(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", use_sentence_transformers=False)
                    q_vec = embedder.encode([query], batch_size=1, normalize=True)
                    query_embedding = ensure_query_shape(q_vec, doc_embeddings.shape[1])
            else:
                embedder = get_embedder(model_name, backend)
                q_vec = embedder.encode([query], batch_size=1, normalize=True)
                query_embedding = ensure_query_shape(q_vec, doc_embeddings.shape[1])

            if query_embedding is None:
                return
            # after:
            # Run hybrid search
            scores, indices, result_metadata = vector_store.hybrid_search(
                query, query_embedding, top_k=top_k, use_reranking=use_reranking
            )

            # ✅ FlashRank rerank step
            if use_flashrank:
                try:
                    # Show top 5 titles before rerank
                    st.write("🔹 Before FlashRank (top 5):", [
                        m.get("title") if isinstance(m, dict) else getattr(m, "title", "") 
                        for m in result_metadata[:5]
                    ])

                    scores, indices, result_metadata = flashrank_rerank(
                        query=query,
                        articles=articles,          # use local list
                        keep_indices=keep_indices,  # use local mapping
                        scores=scores,
                        indices=indices,
                        result_metadata=result_metadata,
                    )

                    # Show top 5 titles after rerank
                    st.write("🔸 After FlashRank (top 5):", [
                        m.get("title") if isinstance(m, dict) else getattr(m, "title", "") 
                        for m in result_metadata[:5]
                    ])

                    st.info("Results reranked with FlashRank ✅")
                except Exception as e:
                    st.warning(f"FlashRank rerank failed: {e}")



        # Display results with enhanced UI
        st.subheader(f"🎯 Top {len(scores)} Results")
        st.info(f"📊 Showing results from {len(texts)} embedded abstracts")

        # Show index statistics
        stats = vector_store.get_index_stats()
        with st.expander("📈 Index Statistics"):
            st.json(stats)

        for rank, (score, idx, meta) in enumerate(zip(scores, indices, result_metadata), start=1):
            if idx < 0 or idx >= len(keep_indices):
                continue
                
            global_idx = keep_indices[idx]
            art = articles[global_idx]

            title = art.title or "Untitled"
            abstract = art.abstract or ""
            abstract_snippet = abstract[:500] + ("…" if len(abstract) > 500 else "")
            url = art.url
            
            # Enhanced metadata display
            meta_parts: List[str] = []
            if art.journal:
                meta_parts.append(f"📚 {art.journal}")
            if art.year:
                meta_parts.append(f"📅 {art.year}")
            if art.authors:
                authors_text = ", ".join(art.authors[:2])
                if len(art.authors) > 2:
                    authors_text += f" et al. ({len(art.authors)} total)"
                meta_parts.append(f"👥 {authors_text}")
            if art.doi:
                meta_parts.append(f"📄 DOI: {art.doi}")
            if getattr(art, 'is_free', False):
                meta_parts.append("🆓 Free full text")

            # Score breakdown
            semantic_score = getattr(meta, 'semantic_score', 0)
            keyword_score = getattr(meta, 'keyword_score', 0)
            
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">
                        <a href="{url}" target="_blank">{title}</a>
                    </div>
                    <div class="result-meta">
                        {' • '.join(meta_parts)}
                    </div>
                    <div class="result-abstract">{abstract_snippet}</div>
                    <div class="score-breakdown">
                        <span class="result-score">Final Score: {float(score):.4f}</span>
                        <br>
                        <span class="metric-badge semantic-badge">Semantic: {semantic_score:.3f}</span>
                        <span class="metric-badge keyword-badge">Keyword: {keyword_score:.3f}</span>
                        <span class="metric-badge rerank-badge">Reranked: {'Yes' if use_reranking else 'No'}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Free full text link (if available)
            if getattr(art, 'is_free', False) and getattr(art, 'full_text_link', None):
                st.markdown(f"Full text: [{art.full_text_link}]({art.full_text_link})")

            # Chat UI removed as per current requirement

        # Enhanced CSV download
        if len(scores) > 0:
            selected = []
            for score, idx, meta in zip(scores, indices, result_metadata):
                if idx < 0 or idx >= len(keep_indices):
                    continue
                global_idx = keep_indices[idx]
                art = articles[global_idx]
                selected.append({
                    "rank": len(selected) + 1,
                    "pmid": art.pmid,
                    "title": art.title,
                    "journal": art.journal or "",
                    "year": art.year or "",
                    "url": art.url,
                    "final_score": float(score),
                    "semantic_score": getattr(meta, 'semantic_score', 0),
                    "keyword_score": getattr(meta, 'keyword_score', 0),
                    "abstract": art.abstract,
                })
            
            if selected:
                df = pd.DataFrame(selected)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"pubmed_enhanced_results_{query.replace(' ', '_')[:30]}.csv",
                    mime="text/csv",
                )

    else:
        st.info("💡 Enter a medical query and press Search to begin your research journey!")

if __name__ == "__main__":
    main()
