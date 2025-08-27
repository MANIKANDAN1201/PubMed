from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

# Import organized modules
from ui_components import (
    render_sidebar,
    render_search_header,
    render_search_input,
    render_sort_controls,
    render_result_card,
)
from cache_utils import cached_fetch_pubmed, _hash_key
from search_service import (
    cached_embeddings_chunked,
    generate_query_embedding,
    build_vector_store,
    perform_hybrid_search,
    apply_flashrank_reranking,
    sort_search_results,
)
from data_export import prepare_results_for_export, generate_csv_filename
from query_processing import expand_query
from summary_cluster import prepare_texts_for_embedding, summarize_top_articles

# Import chatbot functionality
from qa_chatbot import (
    initialize_chat_session,
    render_chatbot_interface,
    check_ollama_status,
    format_abstracts_for_context,
    create_summary_prompt,
    get_ollama_response,
)

# ---------------------------
# Page config and Theming
# ---------------------------
st.set_page_config(
    page_title="PubMed Semantic Search",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Font Awesome for icons
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" integrity="sha512-yH3b4vFqzWYsYH1g1q3h6k4Yb7mS8qG4o1uH2c1uYVYcEoP0q3jz3Lz2yZyV0lR8l1wY2q4k7kWf3Z1y5Zl3g==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    """,
    unsafe_allow_html=True,
)

# Theme state (fixed to dark; toggle removed)
st.session_state.theme = "dark"

def inject_theme_css(theme: str):
    dark = theme == "dark"
    # CSS variables and transitions, plus sidebar/nav styles
    st.markdown(
        f"""
        <style>
        :root {{
          --primary: #2563eb;
          --accent: #06b6d4;
          --success: #22c55e;
          --warning: #f59e0b;
          --error: #ef4444;
          --radius: 12px;
          --shadow: 0 6px 18px rgba(0,0,0,{0.35 if dark else 0.1});
        }}

        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
          transition: background-color .25s ease, color .25s ease, border-color .25s ease, box-shadow .25s ease;
          font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        }}

        /* Body background */
        [data-testid="stAppViewContainer"] {{
          background: {('#0f172a') if dark else ('#f7fafc')};
          background-image: {('radial-gradient(600px 200px at 20% 10%, rgba(37,99,235,.15), transparent), radial-gradient(600px 200px at 80% 0%, rgba(124,58,237,.12), transparent)') if dark else 'none'};
        }}

        /* Fixed top nav bar */
        .top-nav {{
          position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
          display: flex; align-items: center; justify-content: space-between;
          padding: 10px 16px;
          background: {('linear-gradient(180deg, rgba(11,18,36,0.85) 0%, rgba(18,26,46,0.85) 100%)') if dark else ('rgba(255,255,255,0.9)')};
          backdrop-filter: saturate(180%) blur(10px);
          border-bottom: 1px solid {('#27364f') if dark else ('#e5e7eb')};
          box-shadow: var(--shadow);
        }}
        .top-left, .top-center, .top-right {{ display: flex; align-items: center; gap: 10px; }}
        .top-center {{ overflow-x: auto; scrollbar-width: none; -ms-overflow-style: none; }}
        .top-center::-webkit-scrollbar {{ display: none; }}
        .brand {{ color: {('#E6EEFF') if dark else ('#0f172a')}; font-weight: 700; letter-spacing: .2px; }}
        .nav-tab {{
          padding: 6px 12px; border-radius: 999px; cursor: pointer; border: 1px solid transparent;
          color: {('#9DB0D1') if dark else ('#334155')};
          position: relative;
        }}
        .nav-tab.active {{
          background: {('#1b2542') if dark else ('#eef2ff')};
          color: {('#E6EEFF') if dark else ('#1e293b')};
          border-color: {('#27364f') if dark else ('#c7d2fe')};
        }}
        .nav-tab.active::after {{ content: ""; position: absolute; left: 20%; right: 20%; bottom: -8px; height: 3px; background: #6366f1; border-radius: 999px; }}
        .icon-btn {{
          padding: 8px 10px; border-radius: 8px; border: 1px solid {('#27364f') if dark else ('#e5e7eb')};
          background: {('#141c2f') if dark else ('#ffffff')}; color: {('#9DB0D1') if dark else ('#334155')};
          transition: transform .12s ease, background-color .2s ease, border-color .2s ease;
        }}
        .icon-btn:hover {{ transform: translateY(-1px); border-color: #6366f1; }}
        @media (max-width: 640px) {{ .brand {{ display: none; }} .top-right {{ gap: 6px; }} }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{
          background: {('#18223a') if dark else ('#ffffff')};
          border-right: 1px solid {('#27364f') if dark else ('#e5e7eb')};
        }}
        [data-testid="stSidebar"] * {{ color: inherit; }}
        
        /* Search container */
        .search-container {{
          margin-top: 70px; /* to offset fixed nav */
          background: {('#0f172a') if dark else ('#ffffff')};
          border: 1px solid {('#27364f') if dark else ('#e5e7eb')};
          border-radius: var(--radius);
          box-shadow: var(--shadow);
          padding: 16px;
        }}

        /* Result card overrides for dark mode */
        .result-card {{
          background: {('linear-gradient(180deg, #111a30 0%, #0f172a 100%)') if dark else ('linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)')};
          border: 1px solid {('#27364f') if dark else ('#e6e6e6')};
          border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: var(--shadow);
        }}
        .result-title a {{ color: {('#E6EEFF') if dark else ('#1f2937')}; text-decoration: none; }}
        .result-meta {{ color: {('#9DB0D1') if dark else ('#6b7280')}; }}
        .result-abstract {{ color: {('#c7d2fe') if dark else ('#374151')}; }}
        .score-breakdown {{ background: {('#10182b') if dark else ('#f3f4f6')}; color: {('#9DB0D1') if dark else ('#4b5563')}; }}
        .metric-badge.semantic-badge {{ background: {('#1e3a8a') if dark else ('#dbeafe')}; color: #c7d2fe; }}
        .metric-badge.keyword-badge {{ background: {('#4b5563') if dark else ('#fef3c7')}; color: {('#E6EEFF') if dark else ('#92400e')}; }}
        .metric-badge.rerank-badge {{ background: {('#064e3b') if dark else ('#dcfce7')}; color: {('#86efac') if dark else ('#166534')}; }}
        .rank-badge {{ background: {('#10182b') if dark else ('#f0f0f0')}; color: {('#9DB0D1') if dark else ('#666')}; padding: 4px 8px; border-radius: 6px; font-size: .8rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme_css(st.session_state.theme)

# ---------------------------
# Top Navigation Bar
# ---------------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Search"

def top_nav():
    active = st.session_state.active_tab
    st.markdown(
        f"""
        <div class="top-nav">
          <div class="top-left">
            <button class="icon-btn" onclick="window.parent.postMessage({{type:'streamlit:toggleSidebar'}}, '*')" title="Toggle sidebar">
              <i class="fa-solid fa-bars"></i>
            </button>
            <div class="brand"><i class="fa-solid fa-microscope" style="margin-right:6px;color:#6366f1"></i>PubMed Semantic Search</div>
          </div>
          <div class="top-center">
            <span class="nav-tab { 'active' if active=='Search' else '' }" onclick="window.dispatchEvent(new CustomEvent('setTab',{{detail:'Search'}}))">Search</span>
            <span class="nav-tab { 'active' if active=='Summary' else '' }" onclick="window.dispatchEvent(new CustomEvent('setTab',{{detail:'Summary'}}))">Summary</span>
            <span class="nav-tab { 'active' if active=='Chatbot' else '' }" onclick="window.dispatchEvent(new CustomEvent('setTab',{{detail:'Chatbot'}}))">Chatbot</span>
          </div>
          <div class="top-right">
            <button class="icon-btn" onclick="window.dispatchEvent(new CustomEvent('clearCache'))" title="Clear cache">
              <i class="fa-solid fa-trash"></i>
            </button>
          </div>
        </div>
        <script>
          window.addEventListener('setTab', (e) => {{
            const frame = window.parent;
            frame.postMessage({{type: 'streamlit:setComponentValue', value: 'setTab:' + e.detail}}, '*');
          }});
          window.addEventListener('clearCache', () => {{
            const frame = window.parent;
            frame.postMessage({{type: 'streamlit:setComponentValue', value: 'clearCache'}}, '*');
          }});
        </script>
        """,
        unsafe_allow_html=True,
    )

top_nav()

# Streamlit-native controls row under the fixed nav (functional)
with st.container():
    st.markdown("<div style='height:56px'></div>", unsafe_allow_html=True)  # spacer for fixed nav
    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        if st.button("Search", use_container_width=True):
            st.session_state.active_tab = "Search"
    with c2:
        if st.button("Summary", use_container_width=True):
            st.session_state.active_tab = "Summary"
    with c3:
        if st.button("Chatbot", use_container_width=True):
            st.session_state.active_tab = "Chatbot"
    with c4:
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear(); st.cache_resource.clear()
            st.success("Cache cleared successfully!")


def main() -> None:
    # Sidebar and settings
    settings = render_sidebar()

    # Tabs simulated via state; render content per active tab
    active = st.session_state.active_tab

    # Search Page
    if active == "Search":
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        render_search_header()
        query, do_search = render_search_input()

        # Reset chat and previous state when new search starts
        if do_search and query.strip():
            # Reset chat and summary for a fresh session
            st.session_state.chat_messages = []
            st.session_state.research_summary = None
            initialize_chat_session()

            # Query expansion
            if settings['expand']:
                try:
                    run_query, synonyms_map, tokens = expand_query(query, email=settings['email'] or "")
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
            email_effective = (settings['email'] or "").strip() or "pubmed-semantic@example.com"

            # Fetch articles
            with st.spinner("📚 Fetching PubMed articles..."):
                try:
                    articles = cached_fetch_pubmed(run_query, settings['retmax'], email_effective, None, settings['free_only'])
                except Exception as e:
                    st.error(f"❌ PubMed request failed: {e}")
                    return

            # Extract texts for embedding
            texts = prepare_texts_for_embedding(articles)
            metadata = []
            keep_indices = []
            for idx, art in enumerate(articles):
                if texts[idx]:
                    keep_indices.append(idx)
                    metadata.append({
                        "pmid": art.pmid,
                        "title": art.title,
                        "journal": art.journal,
                        "year": art.year,
                        "authors": art.authors,
                        "url": art.url,
                        "doi": art.doi
                    })

            if not texts:
                st.warning("⚠️ Fetched articles have no abstracts to embed.")
                return

            # Generate embeddings
            emb_key = _hash_key("embeddings_chunked", run_query, settings['model_name'], settings['backend'], str(settings['retmax']))
            with st.spinner("🧠 Generating embeddings (chunked)..."):
                doc_embeddings = cached_embeddings_chunked(emb_key, texts, settings['model_name'], settings['backend'])

            # Build vector store
            with st.spinner("🔧 Building hybrid search index..."):
                vector_store = build_vector_store(texts, doc_embeddings, metadata)

            # Save index if requested
            if settings['save_index']:
                try:
                    vector_store.save_index(settings['index_name'])
                    st.success(f"💾 Index saved as '{settings['index_name']}'")
                except Exception as e:
                    st.warning(f"⚠️ Failed to save index: {e}")

            # Generate query embedding and perform search
            with st.spinner("🔍 Performing hybrid search..."):
                query_embedding = generate_query_embedding(query, settings['model_name'], settings['backend'], doc_embeddings.shape)
                if query_embedding is None:
                    return
                
                scores, indices, result_metadata = perform_hybrid_search(
                    vector_store, query, query_embedding, settings['top_k'], settings['use_reranking']
                )

                # Apply FlashRank reranking if enabled
                if settings['use_flashrank']:
                    scores, indices, result_metadata = apply_flashrank_reranking(
                        query, articles, keep_indices, scores, indices, result_metadata
                    )

            # Store results in session state
            st.session_state.search_results = {
                'scores': scores,
                'indices': indices,
                'result_metadata': result_metadata,
                'articles': articles,
                'keep_indices': keep_indices,
                'texts': texts,
                'query': query,
                'use_reranking': settings['use_reranking']
            }

        # Results area
        if 'search_results' in st.session_state:
            results = st.session_state.search_results
            st.subheader("Search Results")
            st.info(f"Showing results from {len(results['texts'])} embedded abstracts")

            # Actions row: sort + download
            sort_by, sort_order = render_sort_controls()
            sorted_results = sort_search_results(
                results['scores'], results['indices'], results['result_metadata'],
                results['articles'], results['keep_indices'], sort_by, sort_order
            )
            st.markdown("---")
            for rank, result in enumerate(sorted_results, start=1):
                render_result_card(rank, result["art"], result["score"], result["meta"], results['use_reranking'])

            if len(sorted_results) > 0:
                df = prepare_results_for_export(sorted_results, results['query'])
                if not df.empty:
                    st.download_button(
                        label="Download Results (CSV)",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name=generate_csv_filename(results['query']),
                        mime="text/csv",
                        use_container_width=True,
                        key="download_results_csv",
                    )
        else:
            st.info("💡 Enter a medical query and press Search to begin your research journey!")

        st.markdown('</div>', unsafe_allow_html=True)

    # Summary Page
    elif active == "Summary":
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        st.subheader("Summary")
        if 'search_results' not in st.session_state:
            st.warning("Run a search to enable summarization.")
        else:
            # Controls
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                n_articles = st.slider("Number of articles to summarize", 3, 20, 8)
            with col2:
                ollama_ok = check_ollama_status()
                st.markdown(("✅ Ollama available" if ollama_ok else "❌ Ollama not running"))
            with col3:
                gen = st.button("Generate Summary", type="primary", use_container_width=True)

            # Use cached summary if available
            if st.session_state.get('research_summary'):
                st.markdown("---")
                st.markdown("### 📄 Generated Summary")
                st.markdown(
                    f"""
                    <div style="background:#0f172a; border:1px solid #27364f; border-radius:12px; padding:16px; box-shadow: var(--shadow); color:#E6EEFF; line-height:1.7; font-size:1.02rem;">
                        {st.session_state.research_summary}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if gen:
                results = st.session_state.search_results
                # Build sorted article list by current ranking
                sorted_articles = []
                for score, idx, meta in zip(results['scores'], results['indices'], results['result_metadata']):
                    if 0 <= idx < len(results['keep_indices']):
                        global_idx = results['keep_indices'][idx]
                        sorted_articles.append(results['articles'][global_idx])

                # Prepare context from top N
                context = format_abstracts_for_context(sorted_articles, top_n=n_articles)

                if check_ollama_status():
                    with st.spinner("🤖 Generating AI summary via Ollama..."):
                        prompt = create_summary_prompt(context)
                        summary_text = get_ollama_response(prompt, "llama3.2")
                else:
                    # Fallback to built-in summarizer
                    sorted_for_summary = []
                    for score, idx, meta in zip(results['scores'], results['indices'], results['result_metadata']):
                        if 0 <= idx < len(results['keep_indices']):
                            global_idx = results['keep_indices'][idx]
                            art = results['articles'][global_idx]
                            sorted_for_summary.append({"art": art, "score": float(score), "meta": meta})
                    with st.spinner("🧾 Generating summary..."):
                        summary_text = summarize_top_articles(sorted_for_summary, results['query'], top_n=n_articles)

                st.session_state.research_summary = summary_text
                st.markdown("---")
                st.markdown("### 📄 Generated Summary")
                st.markdown(
                    f"""
                    <div style="background:#0f172a; border:1px solid #27364f; border-radius:12px; padding:16px; box-shadow: var(--shadow); color:#E6EEFF; line-height:1.7; font-size:1.02rem;">
                        {summary_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Clear summary
            if st.button("Clear Summary", use_container_width=True):
                st.session_state.research_summary = None

        st.markdown('</div>', unsafe_allow_html=True)

    # Chatbot Page
    else:  # Chatbot
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        st.subheader("Research Assistant Chatbot")
        disabled = 'search_results' not in st.session_state
        if disabled:
            st.warning("Run a search to enable the chatbot.")
        else:
            # Make current articles available and (optionally) limit context count in session
            st.session_state.current_articles = st.session_state.search_results['articles']
        initialize_chat_session()
        render_chatbot_interface()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
