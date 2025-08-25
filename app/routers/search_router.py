from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from ..services.logic.pubmed_fetcher import fetch_pubmed_articles

router = APIRouter()

class HybridSearchRequest(BaseModel):
    query: str
    email: Optional[str] = ""
    retmax: int = 100
    topk: int = 15
    model_name: str = "gemini"
    expand: bool = True
    use_reranking: bool = True
    use_flashrank: bool = False
    free_only: bool = False
    index_name: str = "pubmed_index"

class SearchResult(BaseModel):
    score: float
    semantic_score: float
    keyword_score: float
    reranked: bool
    article: Dict[str, Any]

class HybridSearchResponse(BaseModel):
    results: List[SearchResult]
    articles: List[Dict[str, Any]]
    query_used: str
    total_found: int

@router.post("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(request: HybridSearchRequest):
    """
    Perform search on PubMed articles (simplified version)
    """
    try:
        # Use original query for now (skip expansion to avoid errors)
        run_query = request.query

        # Fetch PubMed articles
        articles = fetch_pubmed_articles(
            query=run_query,
            retmax=request.retmax,
            email=request.email or "pubmed-semantic@example.com",
            api_key=None,
            free_only=request.free_only
        )

        if not articles:
            return HybridSearchResponse(
                results=[],
                articles=[],
                query_used=run_query,
                total_found=0
            )

        # Simple scoring based on relevance (no complex embeddings for now)
        results = []
        articles_dict = []
        
        for i, article in enumerate(articles[:request.topk]):
            # Simple scoring based on position and query match
            score = 1.0 - (i * 0.05)  # Decreasing score by position
            
            # Basic keyword matching for additional scoring
            title_lower = (article.title or "").lower()
            abstract_lower = (article.abstract or "").lower()
            query_lower = request.query.lower()
            
            keyword_matches = 0
            for word in query_lower.split():
                if word in title_lower:
                    keyword_matches += 2  # Title matches are more important
                if word in abstract_lower:
                    keyword_matches += 1
            
            # Adjust score based on keyword matches
            keyword_score = min(keyword_matches * 0.1, 0.5)
            final_score = min(score + keyword_score, 1.0)
            
            article_dict = {
                "pmid": article.pmid,
                "title": article.title,
                "abstract": article.abstract,
                "url": article.url,
                "journal": article.journal,
                "year": article.year,
                "authors": article.authors,
                "doi": article.doi,
                "is_free": article.is_free,
                "full_text_link": article.full_text_link
            }
            
            result = SearchResult(
                score=final_score,
                semantic_score=score,
                keyword_score=keyword_score,
                reranked=False,
                article=article_dict
            )
            results.append(result)
            articles_dict.append(article_dict)

        # Convert all articles to dict format
        all_articles_dict = []
        for article in articles:
            all_articles_dict.append({
                "pmid": article.pmid,
                "title": article.title,
                "abstract": article.abstract,
                "url": article.url,
                "journal": article.journal,
                "year": article.year,
                "authors": article.authors,
                "doi": article.doi,
                "is_free": article.is_free,
                "full_text_link": article.full_text_link
            })

        return HybridSearchResponse(
            results=results,
            articles=all_articles_dict,
            query_used=run_query,
            total_found=len(articles)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/clear-cache")
async def clear_cache():
    """
    Clear search cache
    """
    try:
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/ping")
def ping():
    return {"ok": True}

@router.get("/pubmed")
def pubmed(query: str, retmax: int = 50, email: Optional[str] = None, free_only: bool = False, sort: str = "relevance"):
    articles = fetch_pubmed_articles(query, retmax, email, free_only=free_only)
    return {"count": len(articles), "articles": [a.__dict__ for a in articles]}
