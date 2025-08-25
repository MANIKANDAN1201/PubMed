from fastapi import APIRouter
from typing import List, Dict
from ..services.summary_service import summarize_articles, cluster_summaries


router = APIRouter()


@router.post("/summarize")
def summarize(payload: dict):
    articles = payload.get("articles", [])
    query = payload.get("query", "")
    return {"summaries": summarize_articles(articles, query)}


@router.post("/cluster")
def cluster(payload: dict):
    articles = payload.get("articles", [])
    query = payload.get("query", "")
    n_clusters = int(payload.get("n_clusters", 3))
    return cluster_summaries(articles, query, n_clusters)


