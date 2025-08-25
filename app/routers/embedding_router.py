from fastapi import APIRouter
from typing import List
from ..services.embedding_service import embed_texts


router = APIRouter()


@router.post("/encode")
def encode(payload: dict):
    texts = payload.get("texts", [])
    model_name = payload.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    use_st = bool(payload.get("use_sentence_transformers", False))
    return {"embeddings": embed_texts(texts, model_name, use_st)}


