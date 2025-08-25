from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

# Import organized modules
from ui_components import setup_page_config, apply_custom_css, render_sidebar, render_search_header, render_search_input, render_sort_controls, render_result_card
from cache_utils import cached_fetch_pubmed, _hash_key
from search_service import cached_embeddings_chunked, generate_query_embedding, build_vector_store, perform_hybrid_search, apply_flashrank_reranking, sort_search_results
from data_export import prepare_results_for_export, generate_csv_filename
from query_processing import expand_query
from summary_cluster import prepare_texts_for_embedding

# Import chatbot functionality
from qa_chatbot import initialize_chat_session, render_chatbot_interface

# Setup page configuration
setup_page_config()
apply_custom_css()


def main() -> None:
    st.title("🔬 PubMed Semantic Search (Enhanced)")
    st.caption("Advanced biomedical literature search with hybrid semantic + keyword search, persistence, and intelligent reranking.")

    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Render search header and input
    render_search_header()
    query, do_search = render_search_input()

    if do_search and query.strip():
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

    # Display results if available in session state
    if 'search_results' in st.session_state:
        results = st.session_state.search_results
        
        # Display results with enhanced UI
        st.subheader(f"Search Results")
        st.info(f"Showing results from {len(results['texts'])} embedded abstracts")

        # Render sort controls
        sort_by, sort_order = render_sort_controls()

        # Sort results
        sorted_results = sort_search_results(
            results['scores'], results['indices'], results['result_metadata'],
            results['articles'], results['keep_indices'], sort_by, sort_order
        )

        st.markdown("---")
        st.subheader("Search Results")

        # Display sorted results
        for rank, result in enumerate(sorted_results, start=1):
            render_result_card(rank, result["art"], result["score"], result["meta"], results['use_reranking'])

        # Enhanced CSV download
        if len(sorted_results) > 0:
            df = prepare_results_for_export(sorted_results, results['query'])
            if not df.empty:
                st.download_button(
                    label="Download Results (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=generate_csv_filename(results['query']),
                    mime="text/csv",
                    use_container_width=True
                )

        # Initialize chat session
        initialize_chat_session()
        
        # Store current articles for chatbot
        st.session_state.current_articles = results['articles']
        
        # Chatbot Interface
        st.markdown("---")
        st.subheader("Research Assistant Chatbot")
        
        render_chatbot_interface()

    else:
        st.info("💡 Enter a medical query and press Search to begin your research journey!")

if __name__ == "__main__":
    main()
