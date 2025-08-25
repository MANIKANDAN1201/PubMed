from typing import List, Dict, Any
from app.services.logic.summary_cluster import AbstractEmbeddingSummarizer
from app.services.logic.pubmed_fetcher import PubMedArticle


def _dict_to_article(d: Dict[str, Any]) -> PubMedArticle:
    return PubMedArticle(
        pmid=d.get("pmid", ""),
        title=d.get("title", ""),
        abstract=d.get("abstract", ""),
        url=d.get("url", ""),
        journal=d.get("journal"),
        year=d.get("year"),
        authors=d.get("authors"),
        doi=d.get("doi"),
        is_free=bool(d.get("is_free", False)),
        full_text_link=d.get("full_text_link"),
        free_source=d.get("free_source"),
    )


def summarize_articles(articles_dicts: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    articles = [_dict_to_article(a) for a in articles_dicts]
    summarizer = AbstractEmbeddingSummarizer()
    summaries = summarizer.get_top_summaries(articles, query, top_n=len(articles))
    return [
        {
            "pmid": s.article.pmid,
            "summary": s.summary,
            "relevance_score": s.relevance_score,
            "key_points": s.key_points,
            "query_terms_found": s.query_terms_found,
        }
        for s in summaries
    ]


def cluster_summaries(articles_dicts: List[Dict[str, Any]], query: str, n_clusters: int = 3) -> Dict:
    articles = [_dict_to_article(a) for a in articles_dicts]
    summarizer = AbstractEmbeddingSummarizer()
    return summarizer.cluster_and_summarize(articles, query, n_clusters=n_clusters)


