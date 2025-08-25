from pubmed_fetcher import fetch_pubmed_articles
from embeddings import TextEmbedder
from improved_vector_store import ImprovedVectorStore
import numpy as np

def full_search_pipeline(query, retmax=50, email=None, api_key=None, free_only=False, model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=10):
    articles = fetch_pubmed_articles(query=query, retmax=retmax, email=email, api_key=api_key, free_only=free_only)
    texts = [f"{a.title}\n{a.abstract}\n{getattr(a, 'full_text', '')}" if getattr(a, 'is_free', False) and getattr(a, 'full_text', None) else f"{a.title}\n{a.abstract}" for a in articles]
    embedder = TextEmbedder(model_name=model_name, use_sentence_transformers=model_name.startswith("sentence-transformers/"))
    embeddings = embedder.encode(texts, batch_size=16, normalize=True)
    vector_store = ImprovedVectorStore()
    vector_store.build_hybrid_index(texts, embeddings, [a.__dict__ for a in articles], index_type="flat")
    query_embedding = embedder.encode([query], batch_size=1, normalize=True)
    scores, indices, metadata = vector_store.hybrid_search(query, query_embedding, top_k=top_k)
    return {
        "scores": scores.tolist(),
        "indices": indices.tolist(),
        "metadata": metadata,
        "articles": [a.__dict__ for a in articles]
    }
