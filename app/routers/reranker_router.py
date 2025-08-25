from fastapi import APIRouter
from typing import List, Any
from ..services.reranker_service import flashrank_reorder


router = APIRouter()


@router.post("/flashrank")
def flashrank(payload: dict):
    query = payload.get("query", "")
    articles = payload.get("articles", [])
    keep_indices = payload.get("keep_indices", [])
    scores = payload.get("scores", [])
    indices = payload.get("indices", [])
    result_metadata = payload.get("result_metadata", [])
    new_scores, new_indices, new_meta = flashrank_reorder(query, articles, keep_indices, scores, indices, result_metadata)
    return {"scores": new_scores, "indices": new_indices, "result_metadata": new_meta}


