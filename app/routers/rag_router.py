from fastapi import APIRouter
from typing import List, Dict, Any
from ..services.rag_pipeline_service import rag_answer
from ..models.rag_models import RAGRequest, RAGResponse


router = APIRouter()


@router.post("/answer", response_model=RAGResponse)
def answer(req: RAGRequest):
    answer, docs = rag_answer(req.query, req.top_n, req.embedding_model, req.llm_model)
    return RAGResponse(answer=answer, documents=docs)


