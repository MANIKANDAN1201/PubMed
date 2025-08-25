from typing import List, Any, Tuple
try:
    from .logic.reranker_flashrank import flashrank_rerank
except Exception:
    from ...reranker_flashrank import flashrank_rerank


def flashrank_reorder(query: str, articles: List[Any], keep_indices: List[int], scores: List[float], indices: List[int], result_metadata: List[Any]) -> Tuple[List[float], List[int], List[Any]]:
    return flashrank_rerank(query=query, articles=articles, keep_indices=keep_indices, scores=scores, indices=indices, result_metadata=result_metadata)


