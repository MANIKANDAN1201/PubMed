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
        
        # Simple response generation without complex RAG pipeline
        if not request.articles:
            return ChatResponse(
                response="I don't have any articles to reference. Please perform a search first.",
                success=True
            )
        
        # Basic context-aware response
        article_count = len(request.articles)
        sample_titles = [art.get('title', 'Untitled')[:100] for art in request.articles[:3]]
        
        response_text = f"""Based on the {article_count} articles in your search results, I can help answer questions about:

• {sample_titles[0] if len(sample_titles) > 0 else 'Research topics'}
• {sample_titles[1] if len(sample_titles) > 1 else 'Medical findings'}
• {sample_titles[2] if len(sample_titles) > 2 else 'Scientific studies'}

Your question: "{request.message}"

I'm currently in simplified mode. For detailed analysis, please ensure all dependencies are properly configured. I can see you have {article_count} articles to work with."""

        return ChatResponse(
            response=response_text,
            success=True
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"I apologize, but I encountered an error: {str(e)}",
            success=False
        )

@router.get("/ping")
def ping():
    return {"ok": True}
