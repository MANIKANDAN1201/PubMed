from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import logging
import os

# Import routers
from routers import search, qa, summary, index, benchmark

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PubMed Semantic Search API",
    description="Advanced biomedical literature search with AI-powered semantic understanding",
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

# Mount static files for frontend
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Include routers
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(qa.router, prefix="/api/v1", tags=["qa"])
app.include_router(summary.router, prefix="/api/v1", tags=["summary"])
app.include_router(index.router, prefix="/api/v1", tags=["index"])
app.include_router(benchmark.router, prefix="/api/v1", tags=["benchmark"])

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    else:
        return {
            "message": "PubMed Semantic Search API",
            "version": "1.0.0",
            "endpoints": {
                "search": "/api/v1/search",
                "qa": "/api/v1/qa",
                "summary": "/api/v1/summary",
                "index": "/api/v1/index",
                "benchmark": "/api/v1/benchmark"
            },
            "docs": "/docs",
            "frontend": "Frontend not found. Please check the frontend directory."
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PubMed Semantic Search API"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
