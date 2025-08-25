from typing import List, Optional
from app.services.logic.pubmed_fetcher import fetch_pubmed_articles, PubMedArticle


def search_pubmed(query: str, retmax: int = 50, email: Optional[str] = None, api_key: Optional[str] = None, sort: str = "relevance", free_only: bool = False) -> List[PubMedArticle]:
    return fetch_pubmed_articles(query=query, retmax=retmax, email=email, api_key=api_key, sort=sort, free_only=free_only)


