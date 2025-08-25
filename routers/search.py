from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
import time

# Import models
try:
    from models.query import QueryRequest, SearchResponse, ErrorResponse
except ImportError:
    # Fallback for when models are not available
    from pydantic import BaseModel
    class QueryRequest(BaseModel): pass
    class SearchResponse(BaseModel): pass
    class ErrorResponse(BaseModel): pass

# Import services
try:
    from services.pubmed_service import PubMedService
except ImportError:
    PubMedService = None

# Import existing functionality with fallbacks
try:
    from embeddings import TextEmbedder
except ImportError:
    TextEmbedder = None

try:
    from improved_vector_store import ImprovedVectorStore
except ImportError:
    ImprovedVectorStore = None

try:
    from reranker_flashrank import flashrank_rerank
except ImportError:
    flashrank_rerank = None

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
pubmed_service = PubMedService() if PubMedService else None

@router.post("/search", response_model=SearchResponse)
async def search_articles(request: QueryRequest):
    """
    Search PubMed articles with semantic and keyword matching
    
    This endpoint performs a comprehensive search using:
    - Query expansion and processing
    - PubMed API fetching
    - Semantic embedding search
    - Keyword TF-IDF search
    - Intelligent reranking
    """
    # Check if required dependencies are available
    if not all([PubMedService, TextEmbedder, ImprovedVectorStore]):
        raise HTTPException(
            status_code=503,
            detail="Search service dependencies not available. Please install required packages."
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing search request: {request.query}")
        
        # Step 1: Fetch articles from PubMed
        logger.info(f"Fetching up to {request.max_results} articles from PubMed...")
        articles, fetch_time = pubmed_service.search_articles(
            query=request.query,
            max_results=request.max_results,
            free_only=request.free_only,
            email=request.email,
            api_key=request.api_key
        )
        
        if not articles:
            raise HTTPException(
                status_code=404,
                detail="No articles found for the given query"
            )
        
        logger.info(f"✅ Retrieved {len(articles)} articles from PubMed in {fetch_time:.2f}s")
        
        # Step 2: Process articles for vector search
        texts, metadata = pubmed_service.process_articles_for_search(articles)
        
        if not texts:
            raise HTTPException(
                status_code=500,
                detail="Failed to process articles for search"
            )
        
        # Step 3: Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} articles...")
        embedder = TextEmbedder(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        doc_embeddings = embedder.encode(texts, batch_size=32, normalize=True)
        logger.info(f"✅ Generated embeddings for {len(doc_embeddings)} articles")
        
        # Step 4: Build vector store and perform search
        logger.info("Building vector store and performing hybrid search...")
        vector_store = ImprovedVectorStore()
        vector_store.semantic_weight = 0.8
        vector_store.keyword_weight = 0.2
        
        # Build index
        vector_store.build_hybrid_index(
            texts=texts,
            embeddings=doc_embeddings,
            metadata=metadata,
            index_type="flat" if len(texts) < 300 else "ivf"
        )
        
        # Generate query embedding
        query_embedding = embedder.encode([request.query], batch_size=1, normalize=True)
        
        # Perform hybrid search
        scores, indices, result_metadata = vector_store.hybrid_search(
            query=request.query,
            query_embedding=query_embedding,
            top_k=request.max_results,
            use_reranking=request.use_reranking
        )
        
        logger.info(f"✅ Hybrid search completed, found {len(scores)} results")
        
        # Step 5: Apply FlashRank reranking if requested
        if request.use_flashrank and len(articles) > 1 and flashrank_rerank:
            try:
                scores, indices, result_metadata = flashrank_rerank(
                    query=request.query,
                    articles=articles,
                    keep_indices=list(range(len(articles))),
                    scores=scores,
                    indices=indices,
                    result_metadata=result_metadata
                )
                logger.info("Applied FlashRank reranking")
            except Exception as e:
                logger.warning(f"FlashRank reranking failed: {e}")
        
        # Step 6: Convert to response format
        response_data = pubmed_service.convert_to_response_format(
            articles=articles,
            scores=scores,
            indices=indices,
            metadata=result_metadata,
            query=request.query,
            search_time=time.time() - start_time,
            use_reranking=request.use_reranking,
            use_flashrank=request.use_flashrank,
            free_only=request.free_only
        )
        
        logger.info(f"Search completed successfully: {response_data['total_results']} results")
        
        return SearchResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/search/health")
async def search_health_check():
    """Health check for search functionality"""
    dependencies_status = {
        "PubMedService": PubMedService is not None,
        "TextEmbedder": TextEmbedder is not None,
        "ImprovedVectorStore": ImprovedVectorStore is not None,
        "flashrank_rerank": flashrank_rerank is not None
    }
    
    return {
        "status": "healthy" if all(dependencies_status.values()) else "degraded",
        "service": "PubMed Search",
        "timestamp": time.time(),
        "dependencies": dependencies_status
    }

@router.get("/search/stats")
async def get_search_stats():
    """Get search service statistics"""
    return {
        "service": "PubMed Search",
        "status": "operational" if all([PubMedService, TextEmbedder, ImprovedVectorStore]) else "degraded",
        "features": [
            "Query expansion",
            "Semantic search",
            "Keyword search",
            "Hybrid ranking",
            "FlashRank reranking" if flashrank_rerank else "Basic reranking"
        ]
    }
