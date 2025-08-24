"""
Q&A Chatbot Module for PubMed Research Assistant
Handles conversational AI functionality using Ollama models
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from pymed.article import PubMedArticle

# Ollama integration for chatbot
try:
    import ollama
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def format_abstracts_for_context(articles: List[PubMedArticle], top_n: int = 5) -> str:
    """Format top N articles as context for the chatbot."""
    context_parts = []
    for i, article in enumerate(articles[:top_n], 1):
        abstract = article.abstract or "No abstract available"
        title = article.title or "No title available"
        pmid = article.pmid or "Unknown ID"
        
        # Clean and format the abstract
        abstract = abstract.replace('\n', ' ').strip()
        title = title.replace('\n', ' ').strip()
        
        context_parts.append(f"""Article {i} (PMID: {pmid}):
Title: {title}
Abstract: {abstract}
---""")
    
    return "\n".join(context_parts)


def create_chatbot_prompt(context: str, question: str) -> str:
    """Create the prompt for the chatbot."""
    return f"""You are a biomedical research assistant chatbot. 
Use the following retrieved PubMed abstracts as your knowledge base to answer questions about the research findings.

IMPORTANT GUIDELINES:
- Base your answers ONLY on the provided PubMed abstracts
- If the answer cannot be found in the provided context, respond with: 'I could not find a reliable answer in the current knowledge base.'
- Be concise but thorough in your responses
- When referencing specific findings, mention the PMID (PubMed ID) of the source article
- Use scientific language appropriate for biomedical research
- If there are conflicting findings across studies, acknowledge this

Context (PubMed Abstracts):
{context}

Question: {question}

Please provide a clear, evidence-based answer based on the scientific literature provided."""


def create_summary_prompt(context: str) -> str:
    """Create the prompt for generating a summary."""
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


def get_ollama_response(prompt: str, model_name: str = "llama3.2") -> str:
    """Get response from Ollama model."""
    try:
        client = Client(host='http://localhost:11434')
        response = client.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}. Please ensure Ollama is running with: ollama serve"


def check_ollama_status() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        client = Client(host='http://localhost:11434')
        # Try to list models to check connectivity
        client.list()
        return True
    except:
        return False


def initialize_chat_session():
    """Initialize chat session state."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "current_articles" not in st.session_state:
        st.session_state.current_articles = []
    if "research_summary" not in st.session_state:
        st.session_state.research_summary = None


def render_chatbot_interface():
    """Render the complete chatbot interface in a collapsible panel."""
    if not OLLAMA_AVAILABLE:
        st.warning("Ollama is not installed. Install with: `pip install ollama` and ensure Ollama is running.")
        return
    
    # Collapsible chatbot panel
    with st.expander("🤖 Research Assistant Chatbot", expanded=False):
        # Configuration
        top_n_abstracts = st.slider(
            "Number of abstracts to use as context",
            min_value=1,
            max_value=10,
            value=5,
            help="How many top results to include in the chatbot's knowledge base"
        )
        
        # Ollama status indicator
        ollama_status = check_ollama_status()
        if ollama_status:
            st.success("✅ Ollama is running and accessible")
        else:
            st.error("❌ Ollama is not running. Please start Ollama with: `ollama serve`")
        
        # Chat Section
        st.markdown("---")
        st.markdown("### 💬 Ask Questions")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_messages):
                if message["role"] == "user":
                    st.markdown(
                        f"""
                        <div style="background: #e3f2fd; padding: 10px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196f3;">
                            <strong>You:</strong> {message['content']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background: #f3e5f5; padding: 10px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #9c27b0;">
                            <strong>Assistant:</strong> {message['content']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        # Chat input
        user_question = st.text_input(
            "Ask about the research findings...",
            placeholder="e.g., What are the main conclusions? What methods were used?",
            key="chat_input"
        )
        
        if st.button("Send Question", type="primary", use_container_width=True):
            if user_question and st.session_state.current_articles:
                # Add user message to chat
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": user_question
                })
                
                # Prepare context from top N articles
                context = format_abstracts_for_context(
                    st.session_state.current_articles, 
                    top_n_abstracts
                )
                
                # Create prompt
                prompt = create_chatbot_prompt(context, user_question)
                
                # Get response from Ollama
                with st.spinner("Thinking..."):
                    response = get_ollama_response(prompt, "llama3.2")
                
                # Add assistant response to chat
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Rerun to display the new message
                st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    
    # Separate Summary Section
    with st.expander("📋 Research Summary Generator", expanded=False):
        # Ollama status indicator for summary section
        ollama_status = check_ollama_status()
        if ollama_status:
            st.success("✅ Ollama is running and accessible")
        else:
            st.error("❌ Ollama is not running. Please start Ollama with: `ollama serve`")
        
        # Summary configuration
        summary_top_n = st.slider(
            "Number of abstracts for summary",
            min_value=1,
            max_value=10,
            value=5,
            help="How many top results to include in the summary",
            key="summary_slider"
        )
        
        # Display existing summary if available
        if st.session_state.research_summary:
            st.markdown("### 📄 Generated Summary")
            st.markdown(st.session_state.research_summary)
        
        # Generate summary button
        if st.button("Generate Research Summary", type="secondary", use_container_width=True):
            if st.session_state.current_articles:
                # Prepare context from top N articles
                context = format_abstracts_for_context(
                    st.session_state.current_articles, 
                    summary_top_n
                )
                
                # Create summary prompt
                prompt = create_summary_prompt(context)
                
                # Get summary from Ollama
                with st.spinner("Generating summary..."):
                    summary = get_ollama_response(prompt, "llama3.2")
                
                # Store summary in session state
                st.session_state.research_summary = summary
                
                # Rerun to display the summary
                st.rerun()
        
        # Clear summary button
        if st.button("Clear Summary", use_container_width=True):
            st.session_state.research_summary = None
            st.rerun()
