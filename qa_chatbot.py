"""
Q&A Chatbot Module for PubMed Research Assistant
Handles conversational AI functionality using Ollama models
Enhanced responsive UI with query mode, sanitized rendering, and icons
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import html
import re
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
    """Render the chatbot interface with attractive left/right bubbles and icons."""
    if not OLLAMA_AVAILABLE:
        st.warning("Ollama is not installed. Install with: `pip install ollama` and ensure Ollama is running.")
        return

    # Styles for chat bubbles (dark-friendly)
    st.markdown(
        """
        <style>
          .chat-wrap { max-height: 52vh; overflow-y: auto; padding: 8px 4px; background: radial-gradient(600px 200px at 20% 0%, rgba(37,99,235,.08), transparent); border: 1px solid #27364f; border-radius: 12px; }
          .chat-row { display: flex; margin: 8px 0; }
          .chat-row.user { justify-content: flex-end; }
          .chat-row.assistant { justify-content: flex-start; }
          .chat-bubble { max-width: 78%; padding: 12px 14px; border-radius: 14px; line-height: 1.5; box-shadow: 0 6px 18px rgba(0,0,0,.25); }
          .chat-bubble.user { background: linear-gradient(135deg,#1e3a8a,#0ea5e9); color: #e6f0ff; border-top-right-radius: 6px; }
          .chat-bubble.assistant { background: #10182b; color: #dbeafe; border-top-left-radius: 6px; border: 1px solid #27364f; }
          .avatar { width: 34px; height: 34px; border-radius: 999px; display: flex; align-items: center; justify-content: center; margin: 0 8px; background: #0f172a; border: 1px solid #27364f; color: #93c5fd; }
          .avatar.user { background: #0b2a4a; color: #bfdbfe; }
          .msg-inner { display: flex; align-items: flex-end; }
          .welcome { display: flex; align-items: center; gap: 8px; padding: 8px 10px; background: #10182b; border: 1px solid #27364f; border-radius: 10px; color: #c7d2fe; }
          .pill { display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; background:#0f172a; border:1px solid #27364f; color:#a5b4fc; font-size:.88rem; }
          .icon-copy { margin-left: 6px; background: #0f172a; border: 1px solid #27364f; color: #a5b4fc; border-radius: 8px; padding: 6px; cursor: pointer; }
          .icon-copy:hover { border-color: #6366f1; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Helpers
    def sanitize_text(s: str) -> str:
        # escape HTML then linkify urls and preserve newlines
        s = html.escape(s or "")
        url_re = re.compile(r"(https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+)")
        s = url_re.sub(r'<a href="\1" target="_blank">\1</a>', s)
        s = s.replace("\n", "<br>")
        return s

    # Configuration
    default_topn = int(st.session_state.get("context_top_n", 5))
    top_n_abstracts = st.slider(
        "Number of abstracts to use as context",
        min_value=1,
        max_value=10,
        value=default_topn,
        help="How many top results to include in the chatbot's knowledge base",
        key="chat_context_n",
    )

    # Ollama status indicator
    if check_ollama_status():
        st.success("✅ Ollama is running and accessible")
    else:
        st.error("❌ Ollama is not running. Please start Ollama with: `ollama serve`")

    st.markdown("---")
    st.markdown("### 💬 Chat")

    # Welcome hint
    st.markdown("<div class='welcome'><i class='fa-solid fa-bell-concierge'></i> Welcome, let’s unveil your findings</div>", unsafe_allow_html=True)

    # Inline SVG icons (ensures rendering without external icon fonts)
    svg_user = """
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 12c2.761 0 5-2.239 5-5s-2.239-5-5-5-5 2.239-5 5 2.239 5 5 5Z" fill="#bfdbfe"/>
      <path d="M4 22c0-4.418 3.582-8 8-8s8 3.582 8 8" stroke="#93c5fd" stroke-width="2" stroke-linecap="round"/>
    </svg>
    """
    svg_robot = """
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="3" y="8" width="18" height="10" rx="3" fill="#c7d2fe"/>
      <circle cx="9" cy="13" r="1.8" fill="#0f172a"/>
      <circle cx="15" cy="13" r="1.8" fill="#0f172a"/>
      <path d="M12 4v3" stroke="#a5b4fc" stroke-width="2" stroke-linecap="round"/>
    </svg>
    """

    # Messages
    chat_html = ["<div class='chat-wrap'>"]
    for idx, msg in enumerate(st.session_state.chat_messages):
        role = msg.get("role", "assistant")
        content = sanitize_text(msg.get("content", ""))
        if role == "user":
            chat_html.append(
                f"<div class='chat-row user'><div class='msg-inner'><div class='chat-bubble user'>{content}</div><div class='avatar user'>{svg_user}</div></div></div>"
            )
        else:
            copy_svg = """
            <svg width='16' height='16' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg' fill='none' stroke='#a5b4fc' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
              <rect x='9' y='9' width='13' height='13' rx='2' ry='2'></rect>
              <path d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'></path>
            </svg>
            """
            chat_html.append(
                f"<div class='chat-row assistant'><div class='msg-inner'><div class='avatar'>{svg_robot}</div><div class='chat-bubble assistant' id='assistant-msg-{idx}'>{content}</div><button class='icon-copy' data-target='assistant-msg-{idx}' title='Copy'>{copy_svg}</button></div></div>"
            )
    chat_html.append("</div>")
    st.markdown("\n".join(chat_html), unsafe_allow_html=True)
    st.markdown(
        """
        <script>
          (function(){
            const attach = () => {
              document.querySelectorAll('.icon-copy').forEach(btn => {
                if (btn.dataset.bound) return;
                btn.dataset.bound = '1';
                btn.addEventListener('click', async () => {
                  const id = btn.getAttribute('data-target');
                  const el = document.getElementById(id);
                  if (!el) return;
                  try {
                    const text = el.innerText || el.textContent || '';
                    await navigator.clipboard.writeText(text);
                    btn.style.borderColor = '#22c55e';
                    setTimeout(()=>{ btn.style.borderColor = '#27364f'; }, 1200);
                  } catch(e) { console.warn('copy failed', e); }
                });
              });
            };
            if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attach); else attach();
          })();
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Input row
    user_question = st.text_input(
        "Ask about the research findings...",
        placeholder="e.g., What are the main conclusions? What methods were used?",
        key="chat_input",
    )
    cols = st.columns([4,1])
    with cols[1]:
        send = st.button("Submit", type="primary", use_container_width=True)

    if send:
        if user_question and st.session_state.current_articles:
            st.session_state.chat_messages.append({"role": "user", "content": user_question})
            context = format_abstracts_for_context(st.session_state.current_articles, top_n_abstracts)
            prompt = create_chatbot_prompt(context, user_question)
            with st.spinner("Thinking..."):
                response = get_ollama_response(prompt, "llama3.2")
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
