from typing import List, Dict, Any

try:
    from ollama import Client
except Exception:
    Client = None


def format_abstracts_for_context(articles: List[Any], top_n: int = 5) -> str:
    parts = []
    for i, article in enumerate(articles[:top_n], 1):
        abstract = getattr(article, 'abstract', None) or getattr(article, 'abstract', '') or ''
        title = getattr(article, 'title', None) or ''
        pmid = getattr(article, 'pmid', None) or 'Unknown ID'
        abstract = str(abstract).replace('\n', ' ').strip()
        title = str(title).replace('\n', ' ').strip()
        parts.append(f"""Article {i} (PMID: {pmid}):
Title: {title}
Abstract: {abstract}
---""")
    return "\n".join(parts)


def create_chatbot_prompt(context: str, question: str) -> str:
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
    try:
        if Client is None:
            return "Ollama client not available. Install and run Ollama."
        client = Client(host='http://localhost:11434')
        response = client.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}. Please ensure Ollama is running with: ollama serve"


