from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os
import re
import numpy as np

from bs4 import BeautifulSoup 
from langchain_community.vectorstores import FAISS as LCFAISS 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 

def ultra_fast_chunking(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """
    Ultra-fast chunking optimized for PubMed abstracts.
    Uses sentence-based splitting with character fallback.
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # For PubMed abstracts (typically 150-300 words), no chunking needed
    if len(text) <= chunk_size:
        return [text.strip()]
    
    # Try sentence-based splitting first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 1:
        # Single sentence, use character-based chunking
        return make_text_chunks(text, chunk_size, chunk_overlap)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If adding this sentence would exceed chunk size, finalize current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If any chunk is too large, use character-based chunking for that chunk
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            # Fallback to character-based chunking for large chunks
            sub_chunks = make_text_chunks(chunk, chunk_size, chunk_overlap)
            final_chunks.extend(sub_chunks)
    
    return final_chunks if final_chunks else [text.strip()]

def make_text_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """
    Fast character-based chunking with simple sentence boundary detection.
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # For small texts, return as-is
    if len(text) <= chunk_size:
        return [text.strip()]
    
    # Simple character-based chunking with overlap
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to find a sentence boundary within the last 50 characters
        if start > 0 and end < len(text):
            search_start = max(start, end - 50)
            search_text = text[search_start:end]
            
            # Find the last sentence ending
            for i in range(len(search_text) - 1, -1, -1):
                if search_text[i] in '.!?':
                    end = search_start + i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move to next chunk with overlap
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks if chunks else [text.strip()]

def semantic_chunking(text: str, chunk_size: int = 2000, chunk_overlap: int = 300) -> List[str]:
    """
    Fast semantic chunking - just calls ultra_fast_chunking.
    """
    return ultra_fast_chunking(text, chunk_size, chunk_overlap)

def make_text_chunks_legacy(text: str, chunk_size: int = 2000, chunk_overlap: int = 300) -> List[str]:
    """
    Legacy function for compatibility - uses ultra_fast_chunking.
    """
    return ultra_fast_chunking(text, chunk_size, chunk_overlap)


def fetch_full_text(article_url: str) -> Optional[str]:
    """
    Best effort fetch of full text from an article URL. Prioritizes PMC HTML pages.
    Returns plain text or None.
    """
    try:
        if not article_url:
            return None
        # Skip PDFs for now
        if article_url.lower().endswith(".pdf"):
            return None
        resp = requests.get(article_url, timeout=12)
        if resp.status_code != 200 or not resp.text:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        # Prefer article tag if present
        main = soup.find("article") or soup.find(id="maincontent") or soup.body
        if not main:
            main = soup

        # Remove nav, footer, script, style
        for tag in main.find_all(["nav", "footer", "script", "style", "noscript"]):
            tag.extract()

        text = main.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return text if len(text) > 500 else None
    except Exception:
        return None


def build_embeddings(model_choice: str = "gemini", device: Optional[str] = None):
    """
    Returns an embeddings object compatible with LangChain vectorstores.
    - gemini: GoogleGenerativeAIEmbeddings (uses best free model: text-embedding-004)
    - pubmedbert/biobert: HuggingFaceEmbeddings
    """
    lower = (model_choice or "").lower()
    if lower == "gemini":
        # Use the correct Gemini embedding model name
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Map friendly names to HF identifiers
    name_map: Dict[str, str] = {
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "biobert": "dmis-lab/biobert-base-cased-v1.1",
    }
    model_name = name_map.get(lower, model_choice)
    model_kwargs = {"device": device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


def build_vector_store(chunks: List[str], metadatas: Optional[List[Dict]] = None, embedding_model: str = "gemini"):
    embeddings = build_embeddings(embedding_model)
    if metadatas is None:
        metadatas = [{} for _ in chunks]
    return LCFAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)


def make_llm(model: str = "gemini-pro"):
    # Default to Gemini 1.5 Pro
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro")


def retrieve_top_chunks(vector_store, query: str, top_n: int = 6):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_n})
    return retriever.get_relevant_documents(query)


def answer_with_rag(vector_store, query: str, top_n: int = 6, llm_model: str = "gemini-pro") -> Tuple[str, List[Dict]]:
    docs = retrieve_top_chunks(vector_store, query, top_n=top_n)
    llm = make_llm(llm_model)

    # Build prompt
    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        "You are a biomedical assistant. Use ONLY the provided context to answer the question.\n"
        "If the answer is not in the context, say you don't know.\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Cite PMIDs/DOIs if included in the context. Provide a concise, factual answer."
    )
    result = llm.invoke(prompt)
    answer = getattr(result, "content", None) or str(result)
    # Return answer and metadata of retrieved docs
    meta_list: List[Dict] = []
    for d in docs:
        md = dict(d.metadata or {})
        md["snippet"] = d.page_content[:300]
        meta_list.append(md)
    return answer, meta_list


