from typing import List, Dict, Optional, Tuple
import time
import logging
from datetime import datetime

# Import existing functionality with fallbacks
try:
    from pubmed_fetcher import fetch_pubmed_articles, PubMedArticle
except ImportError:
    fetch_pubmed_articles = None
    PubMedArticle = None

try:
    from query_processing import expand_query
except ImportError:
    expand_query = None

logger = logging.getLogger(__name__)

class PubMedService:
    """Service for PubMed article fetching and query processing"""
    
    def __init__(self):
        self.default_email = None
        self.default_api_key = None
        
        # Check if required dependencies are available
        if not fetch_pubmed_articles:
            logger.warning("PubMed fetcher not available - install biopython")
        if not expand_query:
            logger.warning("Query processing not available")
    
    def set_default_credentials(self, email: str, api_key: Optional[str] = None):
        """Set default credentials for PubMed API"""
        self.default_email = email
        self.default_api_key = api_key
    
    def _safe_expand_query(self, query: str) -> str:
        """
        Safely expand query with fallback to original query if expansion fails
        """
        if not expand_query:
            logger.info(f"Using original query (no expansion available): '{query}'")
            return query
        
        try:
            # Try to expand the query
            expanded_query, synonyms_map, tokens = expand_query(query)
            
            # Check if the expanded query is too complex or empty
            if not expanded_query or len(expanded_query) > 1000:
                logger.warning(f"Expanded query too complex, using original: '{query}'")
                return query
            
            # Check for potential problematic characters or patterns
            if expanded_query.count('[') > 20 or expanded_query.count('"') > 50:
                logger.warning(f"Expanded query has too many special characters, using original: '{query}'")
                return query
            
            logger.info(f"Original query: '{query}' -> Expanded: '{expanded_query[:200]}...'")
            return expanded_query
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query: '{query}'")
            return query
    
    def search_articles(
        self,
        query: str,
        max_results: int = 20,
        free_only: bool = False,
        email: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Tuple[List, float]:
        """
        Search PubMed articles with query expansion and processing
        
        Args:
            query: Search query
            max_results: Maximum number of results
            free_only: Whether to return only free articles
            email: Entrez email (uses default if not provided)
            api_key: Entrez API key (uses default if not provided)
            
        Returns:
            Tuple of (articles, search_time)
        """
        if not fetch_pubmed_articles:
            raise Exception("PubMed fetcher not available. Please install biopython.")
        
        start_time = time.time()
        
        try:
            # Use provided credentials or defaults
            effective_email = email or self.default_email
            effective_api_key = api_key or self.default_api_key
            
            # Safely expand query with fallback
            expanded_query = self._safe_expand_query(query)
            
            # Fetch articles from PubMed
            articles = fetch_pubmed_articles(
                query=expanded_query,
                retmax=max_results,
                email=effective_email,
                api_key=effective_api_key,
                free_only=free_only
            )
            
            search_time = time.time() - start_time
            logger.info(f"Found {len(articles)} articles in {search_time:.2f}s")
            
            return articles, search_time
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            
            # If the error is due to query complexity, try with original query
            if "400" in str(e) or "500" in str(e) or "Bad Request" in str(e):
                logger.info("Trying with original query due to API error...")
                try:
                    articles = fetch_pubmed_articles(
                        query=query,  # Use original query
                        retmax=max_results,
                        email=effective_email,
                        api_key=effective_api_key,
                        free_only=free_only
                    )
                    
                    search_time = time.time() - start_time
                    logger.info(f"Found {len(articles)} articles with original query in {search_time:.2f}s")
                    return articles, search_time
                    
                except Exception as e2:
                    logger.error(f"Even original query failed: {e2}")
                    raise Exception(f"PubMed search failed with both expanded and original queries: {str(e2)}")
            
            raise Exception(f"PubMed search failed: {str(e)}")
    
    def process_articles_for_search(
        self,
        articles: List
    ) -> Tuple[List[str], List[Dict]]:
        """
        Process articles for vector search (extract texts and metadata)
        
        Args:
            articles: List of PubMed articles
            
        Returns:
            Tuple of (texts, metadata)
        """
        texts = []
        metadata = []
        
        for i, article in enumerate(articles):
            # Create text for embedding (title + abstract)
            title = getattr(article, 'title', '') or ''
            abstract = getattr(article, 'abstract', '') or ''
            text = f"{title} {abstract}".strip()
            
            if text:
                texts.append(text)
                
                # Create metadata
                meta = {
                    "pmid": getattr(article, 'pmid', f"unknown_{i}"),
                    "title": title,
                    "journal": getattr(article, 'journal', ''),
                    "year": getattr(article, 'year', ''),
                    "authors": getattr(article, 'authors', []),
                    "url": getattr(article, 'url', ''),
                    "doi": getattr(article, 'doi', ''),
                    "abstract": abstract,
                    "is_free": getattr(article, 'is_free', False)
                }
                metadata.append(meta)
        
        logger.info(f"Processed {len(texts)} articles for vector search")
        return texts, metadata
    
    def convert_to_response_format(
        self,
        articles: List,
        scores: List[float],
        indices: List[int],
        metadata: List[Dict],
        query: str,
        search_time: float,
        use_reranking: bool,
        use_flashrank: bool,
        free_only: bool
    ) -> Dict:
        """
        Convert search results to API response format
        
        Args:
            articles: Original PubMed articles
            scores: Search scores
            indices: Result indices
            metadata: Article metadata
            query: Original query
            search_time: Search execution time
            use_reranking: Whether reranking was used
            use_flashrank: Whether FlashRank was used
            free_only: Whether only free articles were requested
            
        Returns:
            Formatted response dictionary
        """
        formatted_articles = []
        
        for rank, (score, idx, meta) in enumerate(zip(scores, indices, metadata), 1):
            if idx < 0 or idx >= len(articles):
                continue
                
            article = articles[idx]
            
            # Extract semantic and keyword scores from metadata
            semantic_score = getattr(meta, 'semantic_score', 0.0)
            keyword_score = getattr(meta, 'keyword_score', 0.0)
            
            formatted_article = {
                "pmid": getattr(article, 'pmid', f"unknown_{idx}"),
                "title": getattr(article, 'title', "No title available"),
                "abstract": getattr(article, 'abstract', "No abstract available"),
                "journal": getattr(article, 'journal', ''),
                "year": getattr(article, 'year', ''),
                "authors": getattr(article, 'authors', []),
                "doi": getattr(article, 'doi', ''),
                "url": getattr(article, 'url', ''),
                "is_free": getattr(article, 'is_free', False),
                "final_score": float(score),
                "semantic_score": float(semantic_score),
                "keyword_score": float(keyword_score),
                "rank": rank
            }
            formatted_articles.append(formatted_article)
        
        response = {
            "query": query,
            "total_results": len(formatted_articles),
            "articles": formatted_articles,
            "search_time": search_time,
            "timestamp": datetime.now().isoformat(),
            "use_reranking": use_reranking,
            "use_flashrank": use_flashrank,
            "free_only": free_only
        }
        
        return response
    
    def get_article_by_pmid(self, pmid: str) -> Optional:
        """
        Get a single article by PMID
        
        Args:
            pmid: PubMed ID
            
        Returns:
            PubMed article or None if not found
        """
        if not fetch_pubmed_articles:
            logger.error("PubMed fetcher not available")
            return None
            
        try:
            articles = fetch_pubmed_articles(
                query=f"pmid:{pmid}",
                retmax=1,
                email=self.default_email,
                api_key=self.default_api_key
            )
            
            if articles:
                return articles[0]
            return None
            
        except Exception as e:
            logger.error(f"Error fetching article {pmid}: {e}")
            return None
