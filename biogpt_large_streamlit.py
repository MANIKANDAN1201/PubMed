import streamlit as st
import numpy as np
from pubmed_fetcher import fetch_pubmed_articles, PubMedArticle
from query_processing import expand_query
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import faiss

# BioGPT-Large embedding utility
class BioGPTEmbedder:
    def __init__(self, model_name: str = "microsoft/BioGPT-Large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def embed(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = embeddings.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-12)

st.set_page_config(page_title="BioGPT-Large Semantic Search", layout="wide")
st.title("🔬 BioGPT-Large Semantic Search")
st.caption("Retrieve relevant PubMed articles using BioGPT-Large embeddings.")

with st.sidebar:
    retmax = st.slider("Articles to fetch", 10, 100, 30)
    top_k = st.slider("Top K Results", 1, 20, 10)
    st.divider()
    st.info("Using BioGPT-Large for embedding.")

query = st.text_input("Enter your medical query", placeholder="e.g., heart attack treatment")
do_search = st.button("Search", type="primary")

if do_search and query.strip():
    st.write(f"**Query:** {query}")
    st.write("Expanding query with medical synonyms and MeSH terms...")
    expanded_query, synonyms_map, tokens = expand_query(query)
    st.write(f"**Expanded Query:** {expanded_query}")

    st.write("Fetching PubMed articles...")
    articles: List[PubMedArticle] = fetch_pubmed_articles(expanded_query, retmax=retmax)
    if not articles:
        st.warning("No articles found for this query.")
    else:
        st.success(f"Fetched {len(articles)} articles.")
        abstracts = [art.abstract for art in articles if art.abstract]
        pmids = [art.pmid for art in articles if art.abstract]
        titles = [art.title for art in articles if art.abstract]
        journals = [art.journal for art in articles if art.abstract]
        years = [art.year for art in articles if art.abstract]
        urls = [art.url for art in articles if art.abstract]
        # Embed all abstracts
        st.write("Embedding abstracts and building FAISS index...")
        embedder = BioGPTEmbedder()
        abs_embeddings = embedder.embed(abstracts)
        dim = abs_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(abs_embeddings)
        # Embed query
        query_embedding = embedder.embed([query])[0].reshape(1, -1)
        scores, indices = faiss_index.search(query_embedding, top_k)
        st.subheader("Top Relevant Articles (Semantic Similarity)")
        for rank, idx in enumerate(indices[0]):
            st.markdown(f"<div style='border:1px solid #eee; border-radius:8px; padding:12px; margin-bottom:10px;'>"
                        f"<b>{titles[idx]}</b> <br>"
                        f"<span style='color:#888;'>Journal: {journals[idx]} | Year: {years[idx]} | PMID: {pmids[idx]}</span><br>"
                        f"<b>Cosine Similarity Score:</b> {scores[0][rank]:.4f}<br>"
                        f"<a href='{urls[idx]}' target='_blank'>View on PubMed</a>"
                        f"</div>", unsafe_allow_html=True)
        if len(indices[0]) == 0:
            st.info("No relevant articles found.")
else:
    st.info("Enter a query and press Search to begin.")
