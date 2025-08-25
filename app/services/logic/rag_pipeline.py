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
    if not text or len(text.strip()) == 0:
        return []
    if len(text) <= chunk_size:
        return [text.strip()]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return make_text_chunks(text, chunk_size, chunk_overlap)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            sub_chunks = make_text_chunks(chunk, chunk_size, chunk_overlap)
            final_chunks.extend(sub_chunks)
    return final_chunks if final_chunks else [text.strip()]


def make_text_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    if not text or len(text.strip()) == 0:
        return []
    if len(text) <= chunk_size:
        return [text.strip()]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if start > 0 and end < len(text):
            search_start = max(start, end - 50)
            search_text = text[search_start:end]
            for i in range(len(search_text) - 1, -1, -1):
                if search_text[i] in '.!?':
                    end = search_start + i + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start >= len(text):
            break
    return chunks if chunks else [text.strip()]


def build_embeddings(model_choice: str = "gemini", device: Optional[str] = None):
    lower = (model_choice or "").lower()
    if lower == "gemini":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro")


def retrieve_top_chunks(vector_store, query: str, top_n: int = 6):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_n})
    return retriever.get_relevant_documents(query)


def answer_with_rag(vector_store, query: str, top_n: int = 6, llm_model: str = "gemini-pro") -> Tuple[str, List[Dict]]:
    docs = retrieve_top_chunks(vector_store, query, top_n=top_n)
    llm = make_llm(llm_model)
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
    meta_list: List[Dict] = []
    for d in docs:
        md = dict(d.metadata or {})
        md["snippet"] = d.page_content[:300]
        meta_list.append(md)
    return answer, meta_list


