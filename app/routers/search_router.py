from fastapi import APIRouter
from typing import Optional
from ..services.pubmed_service import search_pubmed


router = APIRouter()


@router.get("/pubmed")
def pubmed(query: str, retmax: int = 50, email: Optional[str] = None, free_only: bool = False, sort: str = "relevance"):
    articles = search_pubmed(query=query, retmax=retmax, email=email, sort=sort, free_only=free_only)
    return {"count": len(articles), "articles": [a.__dict__ for a in articles]}


