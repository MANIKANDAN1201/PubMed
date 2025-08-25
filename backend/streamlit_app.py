from __future__ import annotations

import hashlib
from typing import Dict, List, Optional
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# Configuration
BACKEND_URL = "http://localhost:8000"
API_BASE = f"{BACKEND_URL}/api"

st.set_page_config(page_title="PubMed Semantic Search (Enhanced)", layout="wide")

# Enhanced styles for better UI (exactly the same as app.py)
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

def check_backend_health():
    """Check if the backend is running and healthy."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException:
        return False, None

def enhanced_search_backend(request_data: dict):
    """Enhanced search using the backend API with full pipeline."""
    try:
        response = requests.post(f"{API_BASE}/search/enhanced", json=request_data, timeout=120)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Enhanced search failed: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"

def expand_query_backend(query: str, expansion_type: str = "medical"):
    """Expand query using the backend API."""
    try:
        payload = {
            "query": query,
            "expansion_type": expansion_type
        }
        
        response = requests.post(f"{API_BASE}/query/expand", json=payload, timeout=10)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Query expansion failed: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"

def enhanced_summarize_articles_backend(query: str, article_pmids: List[str], summary_type: str = "comprehensive"):
    """Enhanced summarization using the backend API."""
    try:
        payload = {
            "query": query,
            "article_pmids": article_pmids,
            "summary_type": summary_type
        }
        
        response = requests.post(f"{API_BASE}/summarize/enhanced", json=payload, timeout=60)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Enhanced summarization failed: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"

def chat_backend(message: str, session_id: str):
    """Chat using the backend API."""
    try:
        payload = {
            "message": message,
            "session_id": session_id
        }
        
        response = requests.post(f"{API_BASE}/chat", json=payload, timeout=10)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Chat failed: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"

@st.cache_data(show_spinner=False)
def _hash_key(*parts: str) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(p.encode("utf-8"))
    return m.hexdigest()

def main() -> None:
    st.title("🔬 PubMed Semantic Search (Enhanced)")
    st.caption("Advanced biomedical literature search with hybrid semantic + keyword search, persistence, and intelligent reranking.")

    # Check backend health
    backend_healthy, health_data = check_backend_health()
    
    if not backend_healthy:
        st.error("⚠️ Backend is not running or not accessible. Please start the FastAPI backend first.")
        st.info("To start the backend, run: `cd backend && python start_backend.py`")
        return

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
        free_only = st.checkbox("Show only FREE full-text articles?", value=False)

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
    st.markdown("---")
    
    # Search header with better styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; text-align: center;">PubMed Semantic Search</h2>
        <p style="color: white; text-align: center; margin: 10px 0 0 0; opacity: 0.9;">Advanced biomedical literature search with AI-powered semantic understanding</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search input with better layout
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input(
            "Enter your medical query", 
            placeholder="e.g., heart attack symptoms, diabetes complications, cancer immunotherapy",
            help="Enter your medical research question or topic of interest"
        )
    with col2:
        do_search = st.button("Search", type="primary", use_container_width=True)
    with col3:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    if do_search and query.strip():
        # Query expansion
        if expand:
            try:
                # Use backend query expansion
                success, expansion_result = expand_query_backend(query, "medical")
                if success:
                    run_query = expansion_result.get("expanded_query", query)
                    st.info(f"🔍 **Expanded query:** {run_query}")
                    with st.expander("🔍 Query Expansion Details", expanded=False):
                        st.write("Original query:", expansion_result.get("original_query", query))
                        st.write("Expanded query:", expansion_result.get("expanded_query", query))
                        st.write("Expansion terms:", expansion_result.get("expansion_terms", []))
                        st.write("Confidence scores:", expansion_result.get("confidence_scores", []))
                else:
                    st.warning(f"⚠️ Query expansion failed: {expansion_result}. Using original query.")
                    run_query = query
            except Exception as e:
                st.warning(f"⚠️ Query expansion failed: {e}. Using original query.")
                run_query = query
        else:
            run_query = query

        # Enhanced search using backend with full pipeline
        with st.spinner("🔬 Performing enhanced search with embeddings and vector store..."):
            try:
                # Prepare enhanced search request
                search_request = {
                    "query": run_query,
                    "max_results": retmax,
                    "top_k": top_k,
                    "embedding_model": model_name,
                    "use_reranking": use_reranking,
                    "use_flashrank": use_flashrank,
                    "free_only": free_only,
                    "expand_query": expand
                }
                
                success, search_result = enhanced_search_backend(search_request)
                if not success:
                    st.error(f"❌ Enhanced search failed: {search_result}")
                    return
                
                # Extract results from backend response
                articles_data = search_result.get("results", [])
                total_fetched = search_result.get("total_fetched", 0)
                total_results = search_result.get("total_results", 0)
                search_time = search_result.get("search_time", 0.0)
                search_type = search_result.get("search_type", "unknown")
                embedding_model = search_result.get("embedding_model", "unknown")
                reranking_applied = search_result.get("reranking_applied", False)
                flashrank_applied = search_result.get("flashrank_applied", False)
                
                st.success(f"✅ Found {total_results} results from {total_fetched} fetched articles in {search_time:.2f} seconds")
                st.info(f"🔧 Search type: {search_type} | Model: {embedding_model}")
                if reranking_applied:
                    st.info("🔄 Intelligent reranking applied")
                if flashrank_applied:
                    st.info("⚡ FlashRank reranking applied")
                
            except Exception as e:
                st.error(f"❌ Enhanced search request failed: {e}")
                return

        if not articles_data:
            st.warning("⚠️ No articles found.")
            return

        # Store results in session state (similar structure to app.py)
        st.session_state.search_results = {
            'scores': [article.get('similarity_score', 0.0) for article in articles_data],
            'indices': list(range(len(articles_data))),
            'result_metadata': articles_data,
            'articles': articles_data,  # Store the full article data
            'keep_indices': list(range(len(articles_data))),
            'texts': [article.get('abstract', '') for article in articles_data],
            'query': query,
            'use_reranking': use_reranking,
            'total_fetched': total_fetched,
            'search_time': search_time,
            'embedding_model': embedding_model,
            'reranking_applied': reranking_applied,
            'flashrank_applied': flashrank_applied
        }

    # Display results if available in session state
    if 'search_results' in st.session_state:
        results = st.session_state.search_results
        
        # Display results with enhanced UI
        st.subheader(f"Search Results")
        st.info(f"Showing {len(results['texts'])} results from {results.get('total_fetched', 0)} fetched articles")
        st.info(f"🔧 Embedding model: {results.get('embedding_model', 'Unknown')} | ⏱️ Search time: {results.get('search_time', 0):.2f}s")

        # Sort controls at the top of results
        st.markdown("### Sort Results")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                options=[
                    "relevance_score",
                    "publication_date", 
                    "journal_name",
                    "title_alphabetical",
                    "semantic_score",
                    "keyword_score"
                ],
                format_func=lambda x: {
                    "relevance_score": "Relevance Score",
                    "publication_date": "Publication Date", 
                    "journal_name": "Journal Name",
                    "title_alphabetical": "Title (A-Z)",
                    "semantic_score": "Semantic Score",
                    "keyword_score": "Keyword Score"
                }[x],
                key="sort_by"
            )
        
        with col2:
            sort_order = st.selectbox(
                "Order",
                options=["desc", "asc"],
                format_func=lambda x: "Descending" if x == "desc" else "Ascending",
                key="sort_order"
            )

        # Sort results based on user selection
        sorted_results = []
        for score, idx, meta in zip(results['scores'], results['indices'], results['result_metadata']):
            if idx < 0 or idx >= len(results['keep_indices']):
                continue
            global_idx = results['keep_indices'][idx]
            art = results['articles'][global_idx]
            
            # Prepare sorting data
            sort_data = {
                "score": score,
                "idx": idx,
                "meta": meta,
                "art": art,
                "relevance_score": float(score),
                "publication_date": art.get('year', '0'),
                "journal_name": art.get('journal', ''),
                "title_alphabetical": art.get('title', ''),
                "semantic_score": art.get('semantic_score', 0),
                "keyword_score": art.get('keyword_score', 0)
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

        st.markdown("---")
        st.subheader("Search Results")

        # Display sorted results
        for rank, result in enumerate(sorted_results, start=1):
            art = result["art"]
            score = result["score"]
            meta = result["meta"]

            title = art.get('title', 'Untitled')
            abstract = art.get('abstract', '')
            abstract_snippet = abstract[:500] + ("…" if len(abstract) > 500 else "")
            url = art.get('url', '')
            
            # Enhanced metadata display
            meta_parts: List[str] = []
            if art.get('journal'):
                meta_parts.append(f"{art['journal']}")
            if art.get('year'):
                meta_parts.append(f"{art['year']}")
            if art.get('authors'):
                authors = art['authors']
                authors_text = ", ".join(authors[:2])
                if len(authors) > 2:
                    authors_text += f" et al. ({len(authors)} total)"
                meta_parts.append(f"{authors_text}")
            if art.get('doi'):
                meta_parts.append(f"DOI: {art['doi']}")
            if art.get('is_free', False):
                meta_parts.append("🆓 Free full text")

            # Score breakdown
            semantic_score = art.get('semantic_score', 0)
            keyword_score = art.get('keyword_score', 0)
            
            st.markdown(
                f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div class="result-title">
                            <a href="{url}" target="_blank">{title}</a>
                        </div>
                        <div style="background: #f0f0f0; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; color: #666;">
                            #{rank}
                        </div>
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
                        <span class="metric-badge rerank-badge">Reranked: {'Yes' if results['use_reranking'] else 'No'}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Free full text link (if available)
            if art.get('is_free', False) and art.get('full_text_link'):
                st.markdown(f"🆓 **Full text available:** [{art['full_text_link']}]({art['full_text_link']})")
            
            # Add spacing between results
            st.markdown("<br>", unsafe_allow_html=True)

        # Enhanced CSV download
        if len(sorted_results) > 0:
            selected = []
            for rank, result in enumerate(sorted_results, start=1):
                art = result["art"]
                score = result["score"]
                meta = result["meta"]
                selected.append({
                    "rank": rank,
                    "pmid": art.get('pmid', ''),
                    "title": art.get('title', ''),
                    "journal": art.get('journal', ''),
                    "year": art.get('year', ''),
                    "url": art.get('url', ''),
                    "final_score": float(score),
                    "semantic_score": art.get('semantic_score', 0),
                    "keyword_score": art.get('keyword_score', 0),
                    "abstract": art.get('abstract', ''),
                    "is_free": art.get('is_free', False),
                    "full_text_link": art.get('full_text_link', ''),
                })
            if selected:
                df = pd.DataFrame(selected)
                st.download_button(
                    label="Download Results (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"pubmed_enhanced_results_{results['query'].replace(' ', '_')[:30]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # Store current articles for chatbot and summarization
        st.session_state.current_articles = results['articles']
        
        # Summarization Interface
        st.markdown("---")
        st.subheader("📝 Article Summarization")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            summary_query = st.text_input(
                "What would you like to know about these articles?",
                value=results['query'],
                placeholder="e.g., What are the main findings about diabetes treatment?",
                help="Enter a specific question to generate a focused summary"
            )
        
        with col2:
            summary_type = st.selectbox(
                "Summary Type",
                options=["comprehensive", "individual", "cluster"],
                help="Choose the type of summary to generate"
            )
        
        # Select articles for summarization
        if len(sorted_results) > 0:
            selected_pmids = st.multiselect(
                "Select Articles for Summarization (PMIDs)",
                options=[art.get('pmid') for art in sorted_results[:10]],  # Show first 10
                default=[art.get('pmid') for art in sorted_results[:3]],  # Default to first 3
                help="Choose which articles to include in the summary"
            )
            
            if st.button("📝 Generate Summary", type="primary"):
                if summary_query and selected_pmids:
                    with st.spinner("📝 Generating enhanced summary..."):
                        success, summary_result = enhanced_summarize_articles_backend(
                            summary_query, selected_pmids, summary_type
                        )
                        
                        if success:
                            st.success(f"✅ Summary generated in {summary_result['processing_time']:.2f} seconds")
                            
                            # Display summary
                            st.markdown("### 📋 Generated Summary")
                            st.markdown(f"**Query:** {summary_result['query']}")
                            st.markdown(f"**Type:** {summary_result['summary_type']}")
                            st.markdown(f"**Relevance Score:** {summary_result['relevance_score']:.3f}")
                            
                            st.markdown("**Summary:**")
                            st.markdown(summary_result['summary'])
                            
                            if summary_result['key_points']:
                                st.markdown("**Key Points:**")
                                for point in summary_result['key_points']:
                                    st.markdown(f"• {point}")
                            
                            # Store in session state
                            st.session_state.last_summary = summary_result
                        else:
                            st.error(f"❌ {summary_result}")
                else:
                    st.warning("⚠️ Please enter a summary query and select articles.")
        
        # Chatbot Interface
        st.markdown("---")
        st.subheader("🤖 Research Assistant Chatbot")
        
        # Simple chat interface using backend
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about the research results..."):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response from backend
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    session_id = "streamlit_session"
                    success, chat_response = chat_backend(prompt, session_id)
                    
                    if success:
                        response_text = chat_response.get("response", "I'm sorry, I couldn't generate a response.")
                    else:
                        response_text = f"Error: {chat_response}"
                    
                    st.write(response_text)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response_text})

    else:
        st.info("💡 Enter a medical query and press Search to begin your research journey!")

if __name__ == "__main__":
    main()

