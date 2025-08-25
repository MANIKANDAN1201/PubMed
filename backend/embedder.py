from embeddings import TextEmbedder
import numpy as np

def embed_articles(articles, model_name):
    texts = [f"{a['title']}\n{a['abstract']}\n{a.get('full_text', '')}" if a.get('is_free') and a.get('full_text') else f"{a['title']}\n{a['abstract']}" for a in articles]
    embedder = TextEmbedder(model_name=model_name, use_sentence_transformers=model_name.startswith("sentence-transformers/"))
    embeddings = embedder.encode(texts, batch_size=16, normalize=True)
    return np.array(embeddings)
