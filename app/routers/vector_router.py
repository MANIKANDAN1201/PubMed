from fastapi import APIRouter
from typing import List, Dict
from ..services.vector_store_service import build_index, search, save, load, stats
from ..models.vector_models import BuildIndexRequest, SearchRequest


router = APIRouter()


@router.post("/build")
def build(req: BuildIndexRequest):
    return build_index(req.texts, req.embeddings, req.metadata, req.index_type or "ivf")


@router.post("/search")
def hybrid_search(req: SearchRequest):
    return search(req.query, req.query_embedding, req.top_k or 10, req.use_reranking if req.use_reranking is not None else True)


@router.post("/save/{name}")
def save_index(name: str):
    return save(name)


@router.post("/load/{name}")
def load_index(name: str):
    return load(name)


@router.get("/stats")
def get_stats():
    return stats()


