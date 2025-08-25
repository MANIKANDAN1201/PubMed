from fastapi import APIRouter, HTTPException
import logging
import time

# Import models
try:
    from models.qa import QARequest, QAResponse, SummaryRequest, SummaryResponse
except ImportError:
    from pydantic import BaseModel
    class QARequest(BaseModel): pass
    class QAResponse(BaseModel): pass
    class SummaryRequest(BaseModel): pass
    class SummaryResponse(BaseModel): pass

# Import services
try:
    from services.qa_service import QAService
except ImportError:
    QAService = None

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
qa_service = QAService() if QAService else None

@router.post("/qa", response_model=QAResponse)
async def qa_endpoint(request: QARequest):
    """
    Q&A endpoint using retrieved articles as context
    
    This endpoint:
    - Takes a question and list of articles
    - Uses the articles as context for the LLM
    - Generates an answer based on the research context
    """
    if not qa_service:
        raise HTTPException(
            status_code=503,
            detail="QA service not available. Please install required dependencies."
        )
    
    if not qa_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="QA service dependencies not available. Please check Ollama installation."
        )
    
    try:
        logger.info(f"Processing QA request: {request.question[:100]}...")
        
        # Generate QA response
        result = qa_service.generate_qa_response(
            question=request.question,
            articles=request.articles,
            max_articles=request.max_articles
        )
        
        if "error" in result and result["error"]:
            raise HTTPException(
                status_code=500,
                detail=f"QA generation failed: {result['error']}"
            )
        
        logger.info(f"QA response generated successfully in {result['processing_time']:.2f}s")
        
        return QAResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"QA endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"QA endpoint failed: {str(e)}"
        )

@router.post("/summary", response_model=SummaryResponse)
async def summary_endpoint(request: SummaryRequest):
    """
    Summary generation endpoint using retrieved articles
    
    This endpoint:
    - Takes a list of articles
    - Generates a comprehensive summary of the research
    - Provides insights and key findings
    """
    if not qa_service:
        raise HTTPException(
            status_code=503,
            detail="Summary service not available. Please install required dependencies."
        )
    
    if not qa_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="Summary service dependencies not available. Please check Ollama installation."
        )
    
    try:
        logger.info(f"Processing summary request for {len(request.articles)} articles")
        
        # Generate summary
        result = qa_service.generate_summary(
            articles=request.articles,
            max_articles=request.max_articles
        )
        
        if "error" in result and result["error"]:
            raise HTTPException(
                status_code=500,
                detail=f"Summary generation failed: {result['error']}"
            )
        
        logger.info(f"Summary generated successfully in {result['processing_time']:.2f}s")
        
        return SummaryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summary endpoint failed: {str(e)}"
        )

@router.get("/qa/health")
async def qa_health_check():
    """Health check for Q&A functionality"""
    if not qa_service:
        return {
            "status": "not available",
            "service": "Q&A",
            "message": "QA service not initialized"
        }
    
    return {
        "status": "healthy" if qa_service.is_available() else "degraded",
        "service": "Q&A",
        "timestamp": time.time(),
        "available": qa_service.is_available(),
        "supported_models": qa_service.get_supported_models() if qa_service else []
    }

@router.get("/qa/models")
async def get_supported_models():
    """Get list of supported LLM models"""
    if not qa_service:
        return {"models": [], "message": "QA service not available"}
    
    return {
        "models": qa_service.get_supported_models(),
        "current_model": qa_service.model_name
    }
