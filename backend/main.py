"""
FastAPI Backend for PubMed Semantic Search System

This backend provides RESTful APIs for:
- PubMed article search and retrieval
- Semantic search with embeddings
- Article summarization
- Query processing and expansion
- Vector store operations
- Chatbot functionality
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn
import asyncio
import logging
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PubMed system components with error handling
try:
    from pubmed_fetcher import PubMedArticle, fetch_pubmed_articles
    PUBMED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PubMed fetcher not available: {e}")
    PUBMED_AVAILABLE = False
    # Create minimal fallback classes
    from dataclasses import dataclass
    @dataclass
    class PubMedArticle:
        pmid: str
        title: str
        abstract: str
        authors: List[str]
        journal: str
        publication_date: str
        doi: str = ""
    
    def fetch_pubmed_articles(query: str, retmax: int = 20):
        return []

try:
    from embeddings import TextEmbedder
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TextEmbedder not available: {e}")
    EMBEDDINGS_AVAILABLE = False
    TextEmbedder = None

try:
    from improved_vector_store import ImprovedVectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ImprovedVectorStore not available: {e}")
    VECTOR_STORE_AVAILABLE = False
    ImprovedVectorStore = None

try:
    from summary_cluster import AbstractEmbeddingSummarizer, ArticleSummary
    SUMMARIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AbstractEmbeddingSummarizer not available: {e}")
    SUMMARIZER_AVAILABLE = False
    AbstractEmbeddingSummarizer = None
    ArticleSummary = None

try:
    from query_processing import expand_query
    QUERY_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Query processing not available: {e}")
    QUERY_PROCESSING_AVAILABLE = False
    expand_query = None

# Import optional components with error handling
try:
    from rag_pipeline import ultra_fast_chunking
    RAG_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG pipeline not available: {e}")
    RAG_PIPELINE_AVAILABLE = False
    ultra_fast_chunking = None

try:
    from reranker_flashrank import flashrank_rerank
    RERANKER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FlashRank reranker not available: {e}")
    RERANKER_AVAILABLE = False
    flashrank_rerank = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PubMed Semantic Search API",
    description="Backend API for PubMed semantic search, summarization, and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=20, description="Maximum number of results")
    use_semantic_search: bool = Field(default=True, description="Enable semantic search")
    use_keyword_search: bool = Field(default=True, description="Enable keyword search")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model to use")

class EnhancedSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=100, description="Maximum number of results to fetch")
    top_k: int = Field(default=15, description="Top-K results to return")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model to use")
    use_reranking: bool = Field(default=True, description="Enable intelligent reranking")
    use_flashrank: bool = Field(default=False, description="Use FlashRank reranker")
    free_only: bool = Field(default=False, description="Show only free full-text articles")
    expand_query: bool = Field(default=True, description="Expand query with synonyms")

class SummaryRequest(BaseModel):
    query: str = Field(..., description="Query for summarization")
    article_pmids: List[str] = Field(..., description="List of PMIDs to summarize")
    summary_type: str = Field(default="comprehensive", description="Type of summary: comprehensive, individual, or cluster")

class QueryExpansionRequest(BaseModel):
    query: str = Field(..., description="Query to expand")
    expansion_type: str = Field(default="medical", description="Type of expansion: medical, general, or hybrid")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str = Field(..., description="Chat session ID")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[Dict[str, Any]]
    search_time: float
    search_type: str

class EnhancedSearchResponse(BaseModel):
    query: str
    total_fetched: int
    total_results: int
    results: List[Dict[str, Any]]
    search_time: float
    search_type: str
    embedding_model: str
    reranking_applied: bool
    flashrank_applied: bool

class SummaryResponse(BaseModel):
    query: str
    summary: str
    summary_type: str
    relevance_score: float
    key_points: List[str]
    processing_time: float

class QueryExpansionResponse(BaseModel):
    original_query: str
    expanded_query: str
    expansion_terms: List[str]
    confidence_scores: List[float]

class ChatResponse(BaseModel):
    message: str
    session_id: str
    response: str
    timestamp: datetime

# Global instances (initialize on startup)
text_embedder: Optional[TextEmbedder] = None
vector_store: Optional[ImprovedVectorStore] = None
summarizer: Optional[AbstractEmbeddingSummarizer] = None

@app.on_event("startup")
async def startup_event():
    """Initialize global instances on startup."""
    global text_embedder, vector_store, summarizer
    
    logger.info("Initializing PubMed system components...")
    
    # Initialize text embedder if available
    if EMBEDDINGS_AVAILABLE and TextEmbedder:
        try:
            text_embedder = TextEmbedder("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("✓ Text embedder initialized")
        except Exception as e:
            logger.warning(f"⚠️ Text embedder initialization failed: {e}")
            text_embedder = None
    else:
        logger.warning("⚠️ Text embedder not available")
    
    # Initialize vector store if available
    if VECTOR_STORE_AVAILABLE and ImprovedVectorStore:
        try:
            vector_store = ImprovedVectorStore()
            logger.info("✓ Vector store initialized")
        except Exception as e:
            logger.warning(f"⚠️ Vector store initialization failed: {e}")
            vector_store = None
    else:
        logger.warning("⚠️ Vector store not available")
    
    # Initialize summarizer if available
    if SUMMARIZER_AVAILABLE and AbstractEmbeddingSummarizer:
        try:
            summarizer = AbstractEmbeddingSummarizer()
            logger.info("✓ Summarizer initialized")
        except Exception as e:
            logger.warning(f"⚠️ Summarizer initialization failed: {e}")
            summarizer = None
    else:
        logger.warning("⚠️ Summarizer not available")
    
    # Log final status
    available_components = sum([
        text_embedder is not None,
        vector_store is not None,
        summarizer is not None
    ])
    
    if available_components > 0:
        logger.info(f"✅ {available_components}/3 components initialized successfully")
    else:
        logger.warning("⚠️ No components initialized - running in fallback mode")

@app.get("/")
async def root():
    """Root endpoint with system status."""
    return {
        "message": "PubMed Semantic Search API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "text_embedder": text_embedder is not None,
            "vector_store": vector_store is not None,
            "summarizer": summarizer is not None
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "text_embedder": text_embedder is not None,
            "vector_store": vector_store is not None,
            "summarizer": summarizer is not None
        }
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_articles(request: SearchRequest):
    """Search PubMed articles using semantic and keyword search."""
    try:
        start_time = datetime.now()
        
        logger.info(f"Processing search request: {request.query}")
        
        # Fetch articles from PubMed
        articles = fetch_pubmed_articles(request.query, retmax=request.max_results)
        
        if not articles:
            return SearchResponse(
                query=request.query,
                total_results=0,
                results=[],
                search_time=0.0,
                search_type="no_results"
            )
        
        # Prepare results
        results = []
        for article in articles:
            result = {
                "pmid": str(article.pmid) if article.pmid else "No PMID",
                "title": str(article.title) if article.title else "No title",
                "abstract": str(article.abstract) if article.abstract else "No abstract",
                "url": str(article.url) if article.url else "No URL",
                "journal": str(article.journal) if article.journal else "No journal",
                "year": str(article.year) if article.year else "No year",
                "authors": [str(author) for author in (article.authors or [])],
                "doi": str(article.doi) if article.doi else "No DOI",
                "is_free": bool(article.is_free),
                "full_text_link": str(article.full_text_link) if article.full_text_link else None,
                "free_source": str(article.free_source) if article.free_source else None
            }
            results.append(result)
        
        # Calculate search time
        search_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Search completed: {len(results)} results in {search_time:.2f}s")
        
        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=results,
            search_time=search_time,
            search_type="semantic_keyword"
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/search/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Perform semantic search using embeddings."""
    try:
        start_time = datetime.now()
        
        logger.info(f"Processing semantic search: {request.query}")
        
        # Fetch articles
        articles = fetch_pubmed_articles(request.query, retmax=request.max_results)
        
        if not articles:
            return SearchResponse(
                query=request.query,
                total_results=0,
                results=[],
                search_time=0.0,
                search_type="semantic"
            )
        
        # Perform semantic search using embeddings
        if text_embedder and vector_store:
            # Encode query
            query_embedding = text_embedder.encode([request.query])[0]
            
            # Calculate similarities and rank articles
            ranked_articles = []
            for article in articles:
                article_text = f"{article.title}\n{article.abstract}"
                article_embedding = text_embedder.encode([article_text])[0]
                
                # Calculate cosine similarity
                similarity = calculate_cosine_similarity(query_embedding, article_embedding)
                
                ranked_articles.append((article, similarity))
            
            # Sort by similarity score
            ranked_articles.sort(key=lambda x: x[1], reverse=True)
            
            # Prepare results with scores
            results = []
            for article, score in ranked_articles:
                result = {
                    "pmid": str(article.pmid) if article.pmid else "No PMID",
                    "title": str(article.title) if article.title else "No title",
                    "abstract": str(article.abstract) if article.abstract else "No abstract",
                    "url": str(article.url) if article.url else "No URL",
                    "journal": str(article.journal) if article.journal else "No journal",
                    "year": str(article.year) if article.year else "No year",
                    "authors": [str(author) for author in (article.authors or [])],
                    "doi": str(article.doi) if article.doi else "No DOI",
                    "is_free": bool(article.is_free),
                    "full_text_link": str(article.full_text_link) if article.full_text_link else None,
                    "free_source": str(article.free_source) if article.free_source else None,
                    "similarity_score": float(score) if score is not None else 0.0
                }
                results.append(result)
        else:
            # Fallback to basic results
            results = []
            for article in articles:
                result = {
                    "pmid": str(article.pmid) if article.pmid else "No PMID",
                    "title": str(article.title) if article.title else "No title",
                    "abstract": str(article.abstract) if article.abstract else "No abstract",
                    "url": str(article.url) if article.url else "No URL",
                    "journal": str(article.journal) if article.journal else "No journal",
                    "year": str(article.year) if article.year else "No year",
                    "authors": [str(author) for author in (article.authors or [])],
                    "doi": str(article.doi) if article.doi else "No DOI",
                    "is_free": bool(article.is_free),
                    "full_text_link": str(article.full_text_link) if article.full_text_link else None,
                    "free_source": str(article.free_source) if article.free_source else None
                }
                results.append(result)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Semantic search completed: {len(results)} results in {search_time:.2f}s")
        
        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=results,
            search_time=search_time,
            search_type="semantic"
        )
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_articles(request: SummaryRequest):
    """Generate summaries for articles."""
    try:
        start_time = datetime.now()
        
        logger.info(f"Processing summarization request: {request.query}")
        
        if not summarizer:
            # Fallback to basic summarization
            logger.warning("Summarizer not available, using fallback")
            fallback_summary = f"Based on the query '{request.query}', here are the key points from the requested articles. For detailed analysis, please ensure the summarizer component is properly initialized."
            key_points = [f"Article {pmid} is relevant to {request.query}" for pmid in request.article_pmids[:3]]
            
            return SummaryResponse(
                query=request.query,
                summary=fallback_summary,
                summary_type="fallback",
                relevance_score=0.5,
                key_points=key_points,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # Fetch articles by PMIDs
        articles = []
        for pmid in request.article_pmids:
            # For now, we'll create a mock article - in production, you'd fetch from PubMed
            # This is a placeholder - you'll need to implement PMID-based fetching
            mock_article = PubMedArticle(
                pmid=pmid,
                title=f"Article {pmid}",
                abstract=f"Abstract for article {pmid}",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                journal="Mock Journal",
                year="2024",
                authors=["Author 1", "Author 2"],
                doi=f"10.1000/{pmid}"
            )
            articles.append(mock_article)
        
        if not articles:
            raise HTTPException(status_code=400, detail="No articles found for the provided PMIDs")
        
        # Generate summary based on type
        if request.summary_type == "comprehensive":
            # Generate comprehensive summary
            knowledge_chunks = summarizer.create_knowledge_chunks(articles, request.query)
            summary_text = summarizer.generate_comprehensive_summary(knowledge_chunks, request.query)
            
            # Calculate average relevance score
            relevance_score = sum(chunk['relevance_score'] for chunk in knowledge_chunks) / len(knowledge_chunks)
            
            # Extract key points from first article
            key_points = summarizer.extract_relevant_sentences(articles[0].title + "\n" + articles[0].abstract, request.query, max_sentences=3)
            
        elif request.summary_type == "individual":
            # Generate individual summaries
            summaries = summarizer.summarize_all_articles(articles, request.query)
            if summaries:
                summary_text = summaries[0].summary
                relevance_score = summaries[0].relevance_score
                key_points = summaries[0].key_points
            else:
                summary_text = "No summary generated"
                relevance_score = 0.0
                key_points = []
                
        elif request.summary_type == "cluster":
            # Generate cluster summaries
            cluster_summaries = summarizer.cluster_and_summarize(articles, request.query, n_clusters=min(3, len(articles)))
            if cluster_summaries:
                # Use the first cluster summary
                first_cluster = list(cluster_summaries.values())[0]
                summary_text = first_cluster['summary']
                relevance_score = first_cluster['relevance_score']
                key_points = summarizer.extract_relevant_sentences(
                    articles[0].title + "\n" + articles[0].abstract, 
                    request.query, 
                    max_sentences=3
                )
            else:
                summary_text = "No cluster summary generated"
                relevance_score = 0.0
                key_points = []
        else:
            raise HTTPException(status_code=400, detail="Invalid summary type")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Summarization completed in {processing_time:.2f}s")
        
        return SummaryResponse(
            query=request.query,
            summary=summary_text,
            summary_type=request.summary_type,
            relevance_score=relevance_score,
            key_points=key_points,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/api/summarize/enhanced", response_model=SummaryResponse)
async def enhanced_summarize_articles(request: SummaryRequest):
    """Enhanced summarization using actual article data and better processing."""
    try:
        start_time = datetime.now()
        
        logger.info(f"Processing enhanced summarization request: {request.query}")
        
        # Fetch articles by PMIDs
        articles = []
        for pmid in request.article_pmids:
            try:
                # Try to fetch the actual article from PubMed
                article_results = fetch_pubmed_articles(f"pmid:{pmid}", retmax=1)
                if article_results:
                    articles.append(article_results[0])
                else:
                    # Create a mock article if not found
                    mock_article = PubMedArticle(
                        pmid=pmid,
                        title=f"Article {pmid}",
                        abstract=f"Abstract for article {pmid}",
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        journal="Unknown Journal",
                        year="Unknown",
                        authors=["Unknown Author"],
                        doi=f"10.1000/{pmid}"
                    )
                    articles.append(mock_article)
            except Exception as e:
                logger.warning(f"Failed to fetch article {pmid}: {e}")
                # Create a mock article as fallback
                mock_article = PubMedArticle(
                    pmid=pmid,
                    title=f"Article {pmid}",
                    abstract=f"Abstract for article {pmid}",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    journal="Unknown Journal",
                    year="Unknown",
                    authors=["Unknown Author"],
                    doi=f"10.1000/{pmid}"
                )
                articles.append(mock_article)
        
        if not articles:
            raise HTTPException(status_code=400, detail="No articles found for the provided PMIDs")
        
        # Generate summary based on type
        if request.summary_type == "comprehensive":
            # Generate comprehensive summary
            if summarizer:
                try:
                    knowledge_chunks = summarizer.create_knowledge_chunks(articles, request.query)
                    summary_text = summarizer.generate_comprehensive_summary(knowledge_chunks, request.query)
                    
                    # Calculate average relevance score
                    relevance_score = sum(chunk['relevance_score'] for chunk in knowledge_chunks) / len(knowledge_chunks)
                    
                    # Extract key points from first article
                    key_points = summarizer.extract_relevant_sentences(articles[0].title + "\n" + articles[0].abstract, request.query, max_sentences=3)
                except Exception as e:
                    logger.warning(f"Summarizer failed: {e}, using fallback")
                    summary_text, relevance_score, key_points = generate_fallback_summary(articles, request.query)
            else:
                summary_text, relevance_score, key_points = generate_fallback_summary(articles, request.query)
                
        elif request.summary_type == "individual":
            # Generate individual summaries
            if summarizer:
                try:
                    summaries = summarizer.summarize_all_articles(articles, request.query)
                    if summaries:
                        summary_text = summaries[0].summary
                        relevance_score = summaries[0].relevance_score
                        key_points = summaries[0].key_points
                    else:
                        summary_text, relevance_score, key_points = generate_fallback_summary(articles, request.query)
                except Exception as e:
                    logger.warning(f"Individual summarization failed: {e}, using fallback")
                    summary_text, relevance_score, key_points = generate_fallback_summary(articles, request.query)
            else:
                summary_text, relevance_score, key_points = generate_fallback_summary(articles, request.query)
                
        elif request.summary_type == "cluster":
            # Generate cluster summaries
            if summarizer:
                try:
                    cluster_summaries = summarizer.cluster_and_summarize(articles, request.query, n_clusters=min(3, len(articles)))
                    if cluster_summaries:
                        # Use the first cluster summary
                        first_cluster = list(cluster_summaries.values())[0]
                        summary_text = first_cluster['summary']
                        relevance_score = first_cluster['relevance_score']
                        key_points = summarizer.extract_relevant_sentences(
                            articles[0].title + "\n" + articles[0].abstract, 
                            request.query, 
                            max_sentences=3
                        )
                    else:
                        summary_text, relevance_score, key_points = generate_fallback_summary(articles, request.query)
                except Exception as e:
                    logger.warning(f"Cluster summarization failed: {e}, using fallback")
                    summary_text, relevance_score, key_points = generate_fallback_summary(articles, request.query)
            else:
                summary_text, relevance_score, key_points = generate_fallback_summary(articles, request.query)
        else:
            raise HTTPException(status_code=400, detail="Invalid summary type")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Enhanced summarization completed in {processing_time:.2f}s")
        
        return SummaryResponse(
            query=request.query,
            summary=summary_text,
            summary_type=request.summary_type,
            relevance_score=relevance_score,
            key_points=key_points,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Enhanced summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced summarization failed: {str(e)}")

def generate_fallback_summary(articles: List[PubMedArticle], query: str) -> tuple[str, float, List[str]]:
    """Generate a fallback summary when the main summarizer is not available."""
    try:
        # Simple fallback summary based on article content
        titles = [art.title for art in articles if art.title]
        abstracts = [art.abstract for art in articles if art.abstract]
        
        # Combine relevant information
        summary_parts = []
        summary_parts.append(f"Based on the query '{query}', here are the key findings from {len(articles)} research articles:")
        
        if titles:
            summary_parts.append("\nKey Article Titles:")
            for i, title in enumerate(titles[:5], 1):  # Show first 5 titles
                summary_parts.append(f"{i}. {title}")
        
        if abstracts:
            # Extract key sentences from abstracts
            all_text = " ".join(abstracts)
            sentences = all_text.split('.')
            # Simple heuristic: longer sentences often contain more information
            key_sentences = sorted(sentences, key=len, reverse=True)[:3]
            summary_parts.append("\nKey Findings:")
            for sentence in key_sentences:
                if len(sentence.strip()) > 20:  # Only meaningful sentences
                    summary_parts.append(f"• {sentence.strip()}")
        
        summary_text = " ".join(summary_parts)
        
        # Generate key points from titles and abstracts
        key_points = []
        if titles:
            key_points.append(f"Found {len(titles)} relevant articles on {query}")
        if abstracts:
            key_points.append("Abstracts contain detailed research findings")
        key_points.append(f"Total articles analyzed: {len(articles)}")
        
        # Calculate a simple relevance score
        relevance_score = min(0.8, len(articles) * 0.1)  # Cap at 0.8
        
        return summary_text, relevance_score, key_points
        
    except Exception as e:
        logger.error(f"Fallback summary generation failed: {e}")
        return f"Summary generation failed for query '{query}'. Please try again.", 0.0, ["Error in summary generation"]

@app.post("/api/query/expand", response_model=QueryExpansionResponse)
async def expand_query_endpoint(request: QueryExpansionRequest):
    """Expand query using medical terminology and synonyms."""
    try:
        start_time = datetime.now()
        
        logger.info(f"Processing query expansion: {request.query}")
        
        # Use the existing query processing module if available
        if QUERY_PROCESSING_AVAILABLE and expand_query:
            try:
                expanded_query = expand_query(request.query, request.expansion_type)
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
                expanded_query = request.query
        else:
            # Fallback to basic expansion
            logger.warning("Query processing not available, using fallback")
            expanded_query = request.query
        
        # Ensure expanded_query is a string and clean it up
        if not isinstance(expanded_query, str):
            if isinstance(expanded_query, tuple) and len(expanded_query) > 0:
                # Extract the first element if it's a tuple
                expanded_query = str(expanded_query[0])
            else:
                expanded_query = str(expanded_query) if expanded_query else request.query
        
        # Clean up the expanded query and extract meaningful terms
        if expanded_query and expanded_query != request.query:
            # Extract meaningful terms from the expanded query
            import re
            # Extract words that look like medical terms
            medical_terms = re.findall(r'\b[a-zA-Z]+\b', expanded_query.lower())
            # Filter out common words and keep medical terms
            common_words = {'and', 'or', 'all', 'fields', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            medical_terms = [term for term in medical_terms if term not in common_words and len(term) > 2]
            if medical_terms:
                expansion_terms = list(set(medical_terms))  # Remove duplicates
            else:
                expansion_terms = request.query.split()
        else:
            # Fallback to original query terms
            expansion_terms = request.query.split()
        
        confidence_scores = [0.8] * len(expansion_terms)  # Placeholder confidence scores
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Query expansion completed in {processing_time:.2f}s")
        
        return QueryExpansionResponse(
            original_query=request.query,
            expanded_query=expanded_query,
            expansion_terms=expansion_terms,
            confidence_scores=confidence_scores
        )
        
    except Exception as e:
        logger.error(f"Query expansion error: {e}")
        raise HTTPException(status_code=500, detail=f"Query expansion failed: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for Q&A functionality."""
    try:
        start_time = datetime.now()
        
        logger.info(f"Processing chat message: {request.message[:50]}...")
        
        # Simple chat response - you can integrate with your chatbot module
        response = f"Thank you for your message: '{request.message}'. This is a placeholder response. In production, this would integrate with your chatbot system."
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Chat response generated in {processing_time:.2f}s")
        
        return ChatResponse(
            message=request.message,
            session_id=request.session_id,
            response=response,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/api/articles/{pmid}")
async def get_article(pmid: str):
    """Get a specific article by PMID."""
    try:
        logger.info(f"Fetching article: {pmid}")
        
        # For now, return a mock article - implement actual PubMed fetching
        article = {
            "pmid": pmid,
            "title": f"Article Title for PMID {pmid}",
            "abstract": f"This is a mock abstract for article {pmid}. In production, this would fetch the actual article from PubMed.",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "journal": "Mock Journal",
            "year": "2024",
            "authors": ["Author 1", "Author 2"],
            "doi": f"10.1000/{pmid}",
            "is_free": False,
            "full_text_link": None,
            "free_source": None
        }
        
        return article
        
    except Exception as e:
        logger.error(f"Article fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch article: {str(e)}")

@app.get("/api/models")
async def get_available_models():
    """Get available embedding models."""
    return {
        "embedding_models": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "gemini"
        ],
        "summary_models": [
            "comprehensive",
            "individual", 
            "cluster"
        ]
    }

@app.get("/api/debug/test")
async def debug_test():
    """Debug endpoint to test basic functionality."""
    try:
        # Test PubMed fetcher
        test_articles = fetch_pubmed_articles("cancer", retmax=2)
        pubmed_status = f"PubMed fetcher working: {len(test_articles)} articles found"
        
        # Test embeddings if available
        embedding_status = "Text embedder not available"
        if text_embedder:
            try:
                test_embedding = text_embedder.encode(["test query"])
                embedding_status = f"Text embedder working: embedding shape {len(test_embedding)}"
            except Exception as e:
                embedding_status = f"Text embedder error: {str(e)}"
        
        # Test vector store if available
        vector_store_status = "Vector store not available"
        if vector_store:
            vector_store_status = "Vector store working"
        
        return {
            "status": "debug_test_completed",
            "timestamp": datetime.now().isoformat(),
            "pubmed_status": pubmed_status,
            "embedding_status": embedding_status,
            "vector_store_status": vector_store_status,
            "components_available": {
                "text_embedder": text_embedder is not None,
                "vector_store": vector_store is not None,
                "summarizer": summarizer is not None
            }
        }
    except Exception as e:
        logger.error(f"Debug test error: {e}")
        return {
            "status": "debug_test_failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/debug/search-test")
async def debug_search_test():
    """Debug endpoint to test search functionality."""
    try:
        # Test basic search
        test_query = "diabetes"
        articles = fetch_pubmed_articles(test_query, retmax=3)
        
        if not articles:
            return {
                "status": "no_articles_found",
                "query": test_query,
                "message": "No articles found for test query"
            }
        
        # Convert to serializable format
        results = []
        for article in articles:
            result = {
                "pmid": str(article.pmid),
                "title": str(article.title) if article.title else "No title",
                "abstract": str(article.abstract) if article.abstract else "No abstract",
                "url": str(article.url) if article.url else "No URL",
                "journal": str(article.journal) if article.journal else "No journal",
                "year": str(article.year) if article.year else "No year",
                "authors": [str(author) for author in (article.authors or [])],
                "doi": str(article.doi) if article.doi else "No DOI",
                "is_free": bool(article.is_free),
                "full_text_link": str(article.full_text_link) if article.full_text_link else None,
                "free_source": str(article.free_source) if article.free_source else None
            }
            results.append(result)
        
        return {
            "status": "search_test_completed",
            "query": test_query,
            "total_results": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Debug search test error: {e}")
        return {
            "status": "search_test_failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/debug/simple-test")
async def simple_test():
    """Simple test endpoint to verify basic functionality."""
    return {
        "message": "Backend is working!",
        "timestamp": datetime.now().isoformat(),
        "status": "ok"
    }

@app.post("/api/search/enhanced", response_model=EnhancedSearchResponse)
async def enhanced_search(request: EnhancedSearchRequest):
    """Enhanced search with full pipeline: embeddings, vector store, and reranking."""
    try:
        start_time = datetime.now()
        
        logger.info(f"Processing enhanced search: {request.query}")
        
        # Step 1: Query expansion if requested
        run_query = request.query
        if request.expand_query and QUERY_PROCESSING_AVAILABLE and expand_query:
            try:
                expanded_result = expand_query(request.query, "medical")
                if isinstance(expanded_result, tuple) and len(expanded_result) > 0:
                    run_query = str(expanded_result[0])
                else:
                    run_query = str(expanded_result) if expanded_result else request.query
                logger.info(f"Query expanded from '{request.query}' to '{run_query}'")
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}, using original query")
                run_query = request.query
        
        # Step 2: Fetch articles from PubMed
        logger.info("📚 Fetching PubMed articles...")
        articles = fetch_pubmed_articles(run_query, retmax=request.max_results)
        
        if not articles:
                return EnhancedSearchResponse(
                    query=request.query,
                    total_fetched=0,
                    total_results=0,
                    results=[],
                    search_time=(datetime.now() - start_time).total_seconds(),
                    search_type="no_results",
                    embedding_model=request.embedding_model,
                    reranking_applied=False,
                    flashrank_applied=False
                )
        
        # Step 3: Filter for free articles if requested
        if request.free_only:
            articles = [art for art in articles if getattr(art, 'is_free', False)]
            logger.info(f"Filtered to {len(articles)} free articles")
        
        # Step 4: Prepare texts for embedding
        texts = []
        metadata = []
        keep_indices = []
        
        for idx, art in enumerate(articles):
            # Prepare text for embedding (title + abstract)
            text = f"{art.title or ''}\n{art.abstract or ''}"
            if text.strip():
                texts.append(text)
                keep_indices.append(idx)
                metadata.append({
                    "pmid": art.pmid,
                    "title": art.title,
                    "journal": art.journal,
                    "year": art.year,
                    "authors": art.authors,
                    "url": art.url,
                    "doi": art.doi,
                    "is_free": getattr(art, 'is_free', False),
                    "full_text_link": getattr(art, 'full_text_link', None),
                    "free_source": getattr(art, 'free_source', None)
                })
        
        if not texts:
            return EnhancedSearchResponse(
                query=request.query,
                total_fetched=len(articles),
                total_results=0,
                results=[],
                search_time=(datetime.now() - start_time).total_seconds(),
                search_type="no_texts",
                embedding_model=request.embedding_model,
                reranking_applied=False,
                flashrank_applied=False
            )
        
        # Step 5: Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts using {request.embedding_model}")
        
        current_embedder = text_embedder
        if not current_embedder:
            # Fallback: create embedder on demand
            try:
                current_embedder = TextEmbedder(request.embedding_model)
                logger.info("Created text embedder on demand")
            except Exception as e:
                logger.error(f"Failed to create text embedder: {e}")
                # Use fallback approach
                return await semantic_search(SearchRequest(
                    query=request.query,
                    max_results=request.top_k,
                    use_semantic_search=True,
                    use_keyword_search=False,
                    embedding_model=request.embedding_model
                ))
        
        # Generate embeddings
        try:
            doc_embeddings = current_embedder.encode(texts, batch_size=16, normalize=True)
            logger.info(f"Generated embeddings with shape: {doc_embeddings.shape}")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return await semantic_search(SearchRequest(
                query=request.query,
                max_results=request.top_k,
                use_semantic_search=True,
                use_keyword_search=False,
                embedding_model=request.embedding_model
            ))
        
        # Step 6: Build vector store and perform search
        logger.info("Building vector store and performing search")
        
        current_vector_store = vector_store
        if not current_vector_store:
            try:
                current_vector_store = ImprovedVectorStore()
                logger.info("Created vector store on demand")
            except Exception as e:
                logger.error(f"Failed to create vector store: {e}")
                # Fallback to basic search
                return await search_articles(SearchRequest(
                    query=request.query,
                    max_results=request.top_k,
                    use_semantic_search=False,
                    use_keyword_search=True,
                    embedding_model=request.embedding_model
                ))
        
        # Build hybrid index
        try:
            current_vector_store.semantic_weight = 0.8
            current_vector_store.keyword_weight = 0.2
            effective_index_type = "flat" if len(texts) < 300 else "ivf"
            current_vector_store.build_hybrid_index(texts, doc_embeddings, metadata, effective_index_type)
            logger.info(f"Built {effective_index_type} index")
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            # Fallback to basic search
            return await search_articles(SearchRequest(
                query=request.query,
                max_results=request.top_k,
                use_semantic_search=False,
                use_keyword_search=True,
                embedding_model=request.embedding_model
            ))
        
        # Step 7: Generate query embedding and search
        try:
            query_embedding = current_embedder.encode([request.query], batch_size=1, normalize=True)
            
            # Ensure query embedding shape matches
            if query_embedding.shape[1] != doc_embeddings.shape[1]:
                logger.error(f"Query embedding dimension {query_embedding.shape[1]} doesn't match index dimension {doc_embeddings.shape[1]}")
                return await search_articles(SearchRequest(
                    query=request.query,
                    max_results=request.top_k,
                    use_semantic_search=False,
                    use_keyword_search=True,
                    embedding_model=request.embedding_model
                ))
            
            # Perform hybrid search
            scores, indices, result_metadata = current_vector_store.hybrid_search(
                request.query, query_embedding, top_k=request.top_k, use_reranking=request.use_reranking
            )
            
            logger.info(f"Hybrid search completed with {len(scores)} results")
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return await search_articles(SearchRequest(
                query=request.query,
                max_results=request.top_k,
                use_semantic_search=False,
                use_keyword_search=True,
                embedding_model=request.embedding_model
            ))
        
        # Step 8: Apply FlashRank reranking if requested
        flashrank_applied = False
        if request.use_flashrank and RERANKER_AVAILABLE and flashrank_rerank:
            try:
                logger.info("Applying FlashRank reranking")
                scores, indices, result_metadata = flashrank_rerank(
                    query=request.query,
                    articles=articles,
                    keep_indices=keep_indices,
                    scores=scores,
                    indices=indices,
                    result_metadata=result_metadata,
                )
                flashrank_applied = True
                logger.info("FlashRank reranking completed")
            except Exception as e:
                logger.warning(f"FlashRank reranking failed: {e}")
        
        # Step 9: Prepare final results
        results = []
        for score, idx, meta in zip(scores, indices, result_metadata):
            if idx < 0 or idx >= len(keep_indices):
                continue
                
            global_idx = keep_indices[idx]
            art = articles[global_idx]
            
            # Get semantic and keyword scores from metadata
            semantic_score = getattr(meta, 'semantic_score', 0.0)
            keyword_score = getattr(meta, 'semantic_score', 0.0)  # Fallback
            
            result = {
                "pmid": str(art.pmid) if art.pmid else "No PMID",
                "title": str(art.title) if art.title else "No title",
                "abstract": str(art.abstract) if art.abstract else "No abstract",
                "url": str(art.url) if art.url else "No URL",
                "journal": str(art.journal) if art.journal else "No journal",
                "year": str(art.year) if art.year else "No year",
                "authors": [str(author) for author in (art.authors or [])],
                "doi": str(art.doi) if art.doi else "No DOI",
                "is_free": bool(art.is_free),
                "full_text_link": str(art.full_text_link) if art.full_text_link else None,
                "free_source": str(art.free_source) if art.free_source else None,
                "similarity_score": float(score) if score is not None else 0.0,
                "semantic_score": float(semantic_score),
                "keyword_score": float(keyword_score),
                "rank": len(results) + 1
            }
            results.append(result)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Enhanced search completed: {len(results)} results in {search_time:.2f}s")
        
        return EnhancedSearchResponse(
            query=request.query,
            total_fetched=len(articles),
            total_results=len(results),
            results=results,
            search_time=search_time,
            search_type="enhanced_hybrid",
            embedding_model=request.embedding_model,
            reranking_applied=request.use_reranking,
            flashrank_applied=flashrank_applied
        )
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}")

# Utility functions
def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Convert numpy types to native Python types for JSON serialization
    return float(dot_product / (norm1 * norm2))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
