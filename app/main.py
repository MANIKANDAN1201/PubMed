from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import os
from dotenv import load_dotenv

from app.routers import benchmark_router, data_analysis_router, embedding_router, vector_router, chatbot_router, query_router, rag_router, reranker_router, summary_router, search_router


def create_app() -> FastAPI:
    # Load environment from potential locations
    try:
        base_root = Path(__file__).resolve().parents[2]  # .../PubMed
        project_root = base_root.parent  # workspace root
        for env_path in [project_root / ".env", base_root / ".env", Path.cwd() / ".env"]:
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=False)
    except Exception:
        pass

    # Fallback Gemini key if not provided via env
    if not os.environ.get("GOOGLE_API_KEY"):
        # User-provided fallback
        os.environ["GOOGLE_API_KEY"] = "AIzaSyB6V3z_hj_DcyfjiMivSBkkZc5-SNSGODM"

    app = FastAPI(title="PubMed API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(query_router.router, prefix="/api/query", tags=["query"]) 
    app.include_router(search_router.router, prefix="/api/search", tags=["search"]) 
    app.include_router(rag_router.router, prefix="/api/rag", tags=["rag"]) 
    app.include_router(summary_router.router, prefix="/api/summary", tags=["summary"]) 
    app.include_router(chatbot_router.router, prefix="/api/chatbot", tags=["chatbot"]) 
    app.include_router(embedding_router.router, prefix="/api/embeddings", tags=["embeddings"]) 
    app.include_router(vector_router.router, prefix="/api/vector", tags=["vector"]) 
    app.include_router(reranker_router.router, prefix="/api/reranker", tags=["reranker"]) 
    app.include_router(benchmark_router.router, prefix="/api/benchmark", tags=["benchmark"]) 
    app.include_router(data_analysis_router.router, prefix="/api/analysis", tags=["analysis"]) 

    base = Path(__file__).parent
    static_dir = base / "frontend" / "static"
    templates_dir = base / "frontend" / "templates"
    static_dir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_path = templates_dir / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return "<h1>PubMed API</h1><p>Frontend not found. Place index.html under app/frontend/templates/</p>"

    return app


app = create_app()


