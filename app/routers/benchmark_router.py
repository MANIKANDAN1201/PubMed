from fastapi import APIRouter
from ..services.benchmark_service import run_bioasq_benchmark

router = APIRouter()


@router.get("/ping")
def ping():
    return {"ok": True}


@router.post("/bioasq")
def bioasq(payload: dict):
    path = payload.get("path", "BioASQ-train-factoid-6b-full-annotated.json")
    model = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    top_k = int(payload.get("top_k", 5))
    recall, ndcg = run_bioasq_benchmark(path, model, top_k)
    return {"recall_at_k": recall, "ndcg_at_k": ndcg}


