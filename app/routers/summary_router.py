from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
import json

router = APIRouter()

def create_summary_prompt(context: str) -> str:
    return f"""You are a biomedical research assistant. 
Please provide a comprehensive summary of the following PubMed abstracts.

IMPORTANT GUIDELINES:
- Create a well-structured summary covering key findings, methods, and conclusions
- Organize the summary into logical sections (e.g., Background, Methods, Key Findings, Conclusions)
- Highlight common themes and patterns across the studies
- Mention any conflicting findings or limitations
- Reference specific PMIDs when discussing individual studies
- Keep the summary concise but comprehensive

Context (PubMed Abstracts):
{context}

Please provide a structured summary of these research findings."""

def generate_ollama_summary(articles: List[Any]) -> str:
    """
    Generate research summary using Ollama Llama3.2
    """
    if not articles:
        return "No articles provided for summary generation."
    
    try:
        # Prepare context from articles
        context_parts = []
        for article in articles:
            pmid = getattr(article, 'pmid', 'Unknown PMID')
            title = getattr(article, 'title', 'No title')
            abstract = getattr(article, 'abstract', 'No abstract available')
            journal = getattr(article, 'journal', '')
            year = getattr(article, 'year', '')
            
            context_part = f"""
PMID: {pmid}
Title: {title}
Journal: {journal} ({year})
Abstract: {abstract}
---"""
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        prompt = create_summary_prompt(context)
        
        # Call Ollama API
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000
                }
            },
            timeout=120
        )
        
        if ollama_response.status_code == 200:
            response_data = ollama_response.json()
            return response_data.get('response', 'No summary generated')
        else:
            return f"Ollama API error: {ollama_response.status_code} - {ollama_response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama. Please ensure Ollama is running on localhost:11434 with llama3.2 model installed."
    except requests.exceptions.Timeout:
        return "Summary generation timed out. Please try again with fewer articles."
    except Exception as e:
        return f"Error generating summary with Ollama: {str(e)}"

class SummaryRequest(BaseModel):
    articles: List[Dict[str, Any]]
    count: int = 10

class SummaryResponse(BaseModel):
    summary: str
    article_count: int

@router.post("/generate", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """
    Generate research summary from articles using Ollama Llama3.2
    """
    try:
        if not request.articles:
            raise HTTPException(status_code=400, detail="No articles provided")
        
        # Convert dict articles to objects with required attributes
        class ArticleObj:
            def __init__(self, data):
                self.pmid = data.get('pmid', '')
                self.title = data.get('title', '')
                self.abstract = data.get('abstract', '')
                self.journal = data.get('journal', '')
                self.year = data.get('year', '')
                self.authors = data.get('authors', [])
                self.url = data.get('url', '')
                self.doi = data.get('doi', '')
        
        article_objects = [ArticleObj(art) for art in request.articles[:request.count]]
        
        # Generate summary using Ollama
        summary = generate_ollama_summary(article_objects)
        
        return SummaryResponse(
            summary=summary,
            article_count=len(article_objects)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

@router.get("/ping")
def ping():
    return {"ok": True}
