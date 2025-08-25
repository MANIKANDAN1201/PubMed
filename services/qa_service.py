from typing import List, Dict, Optional, Any
import logging
import time

# Import existing functionality with fallbacks
try:
    from qa_chatbot import (
        format_abstracts_for_context,
        create_chatbot_prompt,
        create_summary_prompt,
        get_ollama_response
    )
except ImportError:
    format_abstracts_for_context = None
    create_chatbot_prompt = None
    create_summary_prompt = None
    get_ollama_response = None

logger = logging.getLogger(__name__)

class QAService:
    """Service for Q&A and summary generation using retrieved articles"""
    
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        
        # Check if required functions are available
        if not all([format_abstracts_for_context, create_chatbot_prompt, get_ollama_response]):
            logger.warning("QA chatbot functions not available")
    
    def generate_qa_response(
        self,
        question: str,
        articles: List[Dict],
        max_articles: int = 10
    ) -> Dict[str, Any]:
        """
        Generate Q&A response using retrieved articles
        
        Args:
            question: User question
            articles: List of article dictionaries
            max_articles: Maximum number of articles to use as context
            
        Returns:
            Dictionary with response and metadata
        """
        if not all([format_abstracts_for_context, create_chatbot_prompt, get_ollama_response]):
            return {
                "error": "QA service not available",
                "response": "QA functionality requires qa_chatbot module",
                "articles_used": 0,
                "processing_time": 0
            }
        
        start_time = time.time()
        
        try:
            # Limit articles for context
            context_articles = articles[:max_articles]
            
            # Format abstracts for context
            context = format_abstracts_for_context(context_articles)
            
            # Create chatbot prompt
            prompt = create_chatbot_prompt(context, question)
            
            # Get response from Ollama
            response = get_ollama_response(prompt, model=self.model_name)
            
            processing_time = time.time() - start_time
            
            return {
                "response": response,
                "question": question,
                "articles_used": len(context_articles),
                "processing_time": processing_time,
                "model": self.model_name,
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"QA generation failed: {e}")
            return {
                "error": str(e),
                "response": "Failed to generate response",
                "articles_used": 0,
                "processing_time": time.time() - start_time
            }
    
    def generate_summary(
        self,
        articles: List[Dict],
        max_articles: int = 15
    ) -> Dict[str, Any]:
        """
        Generate summary from retrieved articles
        
        Args:
            articles: List of article dictionaries
            max_articles: Maximum number of articles to summarize
            
        Returns:
            Dictionary with summary and metadata
        """
        if not all([format_abstracts_for_context, create_summary_prompt, get_ollama_response]):
            return {
                "error": "Summary service not available",
                "summary": "Summary functionality requires qa_chatbot module",
                "articles_used": 0,
                "processing_time": 0
            }
        
        start_time = time.time()
        
        try:
            # Limit articles for summary
            summary_articles = articles[:max_articles]
            
            # Format abstracts for context
            context = format_abstracts_for_context(summary_articles)
            
            # Create summary prompt
            prompt = create_summary_prompt(context)
            
            # Get response from Ollama
            summary = get_ollama_response(prompt, model=self.model_name)
            
            processing_time = time.time() - start_time
            
            return {
                "summary": summary,
                "articles_used": len(summary_articles),
                "processing_time": processing_time,
                "model": self.model_name,
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {
                "error": str(e),
                "summary": "Failed to generate summary",
                "articles_used": 0,
                "processing_time": time.time() - start_time
            }
    
    def is_available(self) -> bool:
        """Check if QA service is available"""
        return all([
            format_abstracts_for_context,
            create_chatbot_prompt,
            create_summary_prompt,
            get_ollama_response
        ])
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models"""
        return ["llama3.2", "llama3", "llama2", "mistral"]
