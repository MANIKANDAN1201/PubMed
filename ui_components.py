"""
UI Components and Styling for PubMed Semantic Search
Contains all Streamlit UI styling and component definitions.
"""

import streamlit as st
from typing import Dict, List

# Enhanced styles for better UI
CUSTOM_CSS = """
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
    color: #ffffff; 
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 800; 
    font-size: 1.25rem;
    box-shadow: 0 8px 22px rgba(99,102,241,.35), 0 2px 8px rgba(37,99,235,.25);
    border: 1px solid rgba(99,102,241,.55);
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
"""

# Default models and their display names
DEFAULT_MODELS: Dict[str, str] = {
    "🔬 PubMedBERT (Biomedical)": "pubmedbert",
    "🧬 BioBERT (Biomedical)": "biobert",
    "⚡ MiniLM (Fast & Lightweight)": "minilm",
    "🌐 Gemini (API-based, requires key)": "gemini"
}


def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(page_title="PubMed Semantic Search (Improved)", layout="wide")

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with all controls"""
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
        
        st.subheader("🔍 Search Settings")
        free_only = st.checkbox("Show only FREE full-text articles?", value=False)
        # Always enabled settings
        expand = True  # Always expand query with synonyms
        use_reranking = True  # Always enable intelligent reranking
        use_flashrank = True  # Enable FlashRank reranker by default
        
        clear_cache = st.button("🗑️ Clear all cache", type="primary")
        if clear_cache:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared successfully!")
    
    return {
        'email': email,
        'model_name': model_name,
        'backend': backend,
        'retmax': retmax,
        'top_k': top_k,
        'expand': expand,
        'use_reranking': use_reranking,
        'use_flashrank': use_flashrank,
        'free_only': free_only,
        'save_index': False,  # Default value since we removed the UI control
        'index_name': 'pubmed_index'  # Default value since we removed the UI control
    }

def render_search_header():
    """Render the main search header"""
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0; text-align: center;">PubMed Semantic Search</h2>
        <p style="color: white; text-align: center; margin: 10px 0 0 0; opacity: 0.9;">Advanced biomedical literature search with AI-powered semantic understanding</p>
    </div>
    """, unsafe_allow_html=True)

def render_search_input():
    """Render search input controls"""
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
    
    return query, do_search

def render_sort_controls():
    """Render result sorting controls"""
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
    
    return sort_by, sort_order

def render_result_card(rank: int, art, score: float, meta, use_reranking: bool):
    """Render a single result card"""
    title = art.title or "Untitled"
    abstract = art.abstract or ""
    abstract_snippet = abstract[:500] + ("…" if len(abstract) > 500 else "")
    url = art.url
    
    # Enhanced metadata display
    meta_parts: List[str] = []
    if art.journal:
        meta_parts.append(f"{art.journal}")
    if art.year:
        meta_parts.append(f"{art.year}")
    if art.authors:
        authors_text = ", ".join(art.authors[:2])
        if len(art.authors) > 2:
            authors_text += f" et al. ({len(art.authors)} total)"
        meta_parts.append(f"{authors_text}")
    if art.doi:
        meta_parts.append(f"DOI: {art.doi}")
    if getattr(art, 'is_free', False):
        meta_parts.append("Free full text")

    # Score breakdown
    semantic_score = getattr(meta, 'semantic_score', 0)
    keyword_score = getattr(meta, 'keyword_score', 0)
    
    st.markdown(
        f"""
        <div class="result-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div class="result-title">
                    <a href="{url}" target="_blank">{title}</a>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="result-score" title="Relevance score">{float(score):.4f}</span>
                    <div style="background: #f0f0f0; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; color: #666;">
                        #{rank}
                    </div>
                </div>
            </div>
            <div class="result-meta">
                {' • '.join(meta_parts)}
            </div>
            <div class="result-abstract">{abstract_snippet}</div>
            <div class="score-breakdown">
                <span class="metric-badge rerank-badge">Reranked: {'Yes' if use_reranking else 'No'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Free full text link (if available)
    if getattr(art, 'is_free', False) and getattr(art, 'full_text_link', None):
        st.markdown(f"Full text: [{art.full_text_link}]({art.full_text_link})")
    
    # Add spacing between results
    st.markdown("<br>", unsafe_allow_html=True)
