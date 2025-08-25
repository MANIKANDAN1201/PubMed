from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/summary")
async def summary_redirect_endpoint():
    """Summary endpoint - redirected to QA service"""
    raise HTTPException(
        status_code=308,
        detail="Summary endpoint moved to /api/v1/qa/summary"
    )

@router.get("/summary/health")
async def summary_health_check():
    """Health check for summary functionality"""
    return {
        "status": "redirected",
        "service": "Summary",
        "message": "Summary functionality moved to /api/v1/qa/summary",
        "new_endpoint": "/api/v1/qa/summary"
    }
