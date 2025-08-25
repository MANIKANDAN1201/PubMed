from pubmed_fetcher import fetch_pubmed_articles, PubMedArticle

def fetch_articles(query, retmax=50, email=None, api_key=None, free_only=False):
    articles = fetch_pubmed_articles(query=query, retmax=retmax, email=email, api_key=api_key, free_only=free_only)
    return articles
