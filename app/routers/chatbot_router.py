from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

class ChatMessage(BaseModel):
    content: str
    sender: str  # 'user' or 'bot'

class ChatRequest(BaseModel):
    message: str
    articles: List[Dict[str, Any]]
    conversation_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    success: bool

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Generate chatbot response (simplified version)
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        if not request.articles:
            return ChatResponse(
                response="I don't have any articles to reference. Please perform a search first.",
                success=True,
            )

        # Use Ollama-backed service with llama3.2
        try:
            from app.services.chatbot_service import chat_answer
        except Exception as e:
            return ChatResponse(
                response=f"Chat service unavailable: {e}",
                success=False,
            )

        # Limit context size sensibly (frontend already slices, but keep a safe cap)
        context_articles = request.articles[:10]
        answer = chat_answer(context_articles, request.message, top_n=len(context_articles))

        return ChatResponse(response=answer, success=True)

    except HTTPException:
        raise
    except Exception as e:
        return ChatResponse(
            response=f"I apologize, but I encountered an error: {str(e)}",
            success=False,
        )

@router.get("/ping")
def ping():
    return {"ok": True}
