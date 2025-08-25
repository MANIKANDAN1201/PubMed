from fastapi import APIRouter
from typing import List, Dict
from ..services.query_service import process_query
from ..models.query_models import QueryRequest, QueryExpansionResponse


router = APIRouter()


@router.post("/expand", response_model=QueryExpansionResponse)
def expand(req: QueryRequest):
    enhanced_query, synonyms_map, tokens = process_query(req.query, req.email or "")
    return QueryExpansionResponse(enhanced_query=enhanced_query, synonyms_map=synonyms_map, tokens=tokens)


