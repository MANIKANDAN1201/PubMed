from typing import List, Dict, Any
try:
    from .logic.qa_chatbot import format_abstracts_for_context, create_chatbot_prompt, get_ollama_response, create_summary_prompt
except Exception:
    from ...qa_chatbot import format_abstracts_for_context, create_chatbot_prompt, get_ollama_response, create_summary_prompt


def chat_answer(articles: List[Dict[str, Any]], question: str, top_n: int = 5) -> str:
    # articles are dicts; rebuild a minimal text context from title/abstract
    # The original chatbot expects PubMedArticle objects, but its formatting only uses title/abstract/pmid
    # So we can feed dicts with these keys and rely on attribute access via get()
    class _Article:
        def __init__(self, d: Dict[str, Any]):
            self.pmid = d.get("pmid")
            self.title = d.get("title")
            self.abstract = d.get("abstract")

    objs = [_Article(a) for a in articles]
    context = format_abstracts_for_context(objs, top_n=top_n)
    prompt = create_chatbot_prompt(context, question)
    return get_ollama_response(prompt, "llama3.2")


def generate_summary(articles: List[Dict[str, Any]], top_n: int = 5) -> str:
    class _Article:
        def __init__(self, d: Dict[str, Any]):
            self.pmid = d.get("pmid")
            self.title = d.get("title")
            self.abstract = d.get("abstract")

    objs = [_Article(a) for a in articles]
    context = format_abstracts_for_context(objs, top_n=top_n)
    prompt = create_summary_prompt(context)
    return get_ollama_response(prompt, "llama3.2")


