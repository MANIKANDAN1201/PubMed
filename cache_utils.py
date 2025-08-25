"""
Caching utilities for PubMed Semantic Search
Contains all caching functions and hash utilities.
"""

import hashlib
import streamlit as st
from typing import List, Optional
import numpy as np
from pubmed_fetcher import PubMedArticle, fetch_pubmed_articles

@st.cache_data(show_spinner=False)
def _hash_key(*parts: str) -> str:
    """Generate a hash key from multiple string parts"""
    m = hashlib.sha256()
    for p in parts:
        m.update(p.encode("utf-8"))
    return m.hexdigest()

@st.cache_data(show_spinner=False)
def cached_fetch_pubmed(
    query: str,
    retmax: int,
    email: Optional[str],
    api_key: Optional[str],
    free_only: Optional[bool],
) -> List[PubMedArticle]:
    """Cached PubMed article fetching"""
    return fetch_pubmed_articles(
        query=query,
        retmax=retmax,
        email=email,
        api_key=api_key,
        free_only=free_only,
    )

def clear_all_cache():
    """Clear all Streamlit cache"""
    st.cache_data.clear()
    st.cache_resource.clear()
