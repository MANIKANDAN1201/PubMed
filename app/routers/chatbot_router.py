from fastapi import APIRouter
from typing import List, Dict, Any
from ..services.chatbot_service import chat_answer, generate_summary


router = APIRouter()


@router.post("/ask")
def ask(payload: dict):
    articles = payload.get("articles", [])
    question = payload.get("question", "")
    top_n = int(payload.get("top_n", 5))
    return {"answer": chat_answer(articles, question, top_n)}


@router.post("/summarize")
def summarize(payload: dict):
    articles = payload.get("articles", [])
    top_n = int(payload.get("top_n", 5))
    return {"summary": generate_summary(articles, top_n)}


