from typing import List, Dict, Tuple, Optional
import numpy as np
try:
    from .logic.improved_vector_store import ImprovedVectorStore
except Exception:
    from ...improved_vector_store import ImprovedVectorStore


_store: Optional[ImprovedVectorStore] = None


def get_store() -> ImprovedVectorStore:
    global _store
    if _store is None:
        _store = ImprovedVectorStore()
    return _store


def build_index(texts: List[str], embeddings: List[List[float]], metadata: List[Dict], index_type: str = "ivf") -> Dict:
    store = get_store()
    emb = np.asarray(embeddings, dtype=np.float32)
    store.build_hybrid_index(texts, emb, metadata, index_type=index_type)
    return store.get_index_stats()


def search(query: str, query_embedding: List[float], top_k: int = 10, use_reranking: bool = True):
    store = get_store()
    q = np.asarray(query_embedding, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    scores, indices, meta = store.hybrid_search(query, q, top_k=top_k, use_reranking=use_reranking)
    return {
        "scores": [float(s) for s in scores],
        "indices": [int(i) for i in indices],
        "metadata": meta,
    }


def save(name: str) -> Dict:
    store = get_store()
    store.save_index(name)
    return {"saved": True, "name": name}


def load(name: str) -> Dict:
    store = get_store()
    ok = store.load_index(name)
    return {"loaded": bool(ok), "name": name}


def stats() -> Dict:
    store = get_store()
    return store.get_index_stats()


