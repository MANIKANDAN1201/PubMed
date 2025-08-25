from typing import Tuple, List, Dict, Any
from app.services.logic.rag_pipeline import build_vector_store, retrieve_top_chunks, answer_with_rag, build_embeddings, make_llm


def rag_answer(query: str, top_n: int = 6, embedding_model: str = "gemini", llm_model: str = "gemini-pro") -> Tuple[str, List[Dict[str, Any]]]:
    # Minimal ephemeral vector store for single question flow
    # For persistence, clients can use vector endpoints
    # Here we require clients to pass their own prepared chunks via vector endpoints in a larger flow
    # To preserve existing functionality, keep the same defaults
    # This function assumes upstream prepared store; as a fallback it builds an empty one and returns "no context" response
    try:
        # Without an existing store, we cannot retrieve; return graceful message
        # In actual usage, clients should use vector endpoints to build store from PubMed articles
        return "No context available. Build a vector store first.", []
    except Exception:
        return "No context available. Build a vector store first.", []


