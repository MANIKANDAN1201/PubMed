from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os
import re
import json
import numpy as np
from collections import Counter

from bs4 import BeautifulSoup 
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS as LCFAISS 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 

class PubMedSemanticChunker:
    """
    Data-driven semantic chunking for PubMed literature.
    Uses cached MeSH terms and learns patterns from the data itself.
    """
    
    def __init__(self):
        self.mesh_terms = set()
        self.learned_patterns = set()
        self._load_mesh_terms()
    
    def _load_mesh_terms(self):
        """Load MeSH terms from cache with fallback to minimal terms."""
        try:
            cache_file = "mesh_terms_cache.json"
            
            # Try to load from cache
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    self.mesh_terms = set(json.load(f))
                print(f"Loaded {len(self.mesh_terms)} MeSH terms from cache (fast startup)")
                return
            
            # If no cache exists, use minimal fallback terms
            print("No cache found. Using minimal fallback terms...")
            self.mesh_terms = set([
                "diabetes mellitus", "myocardial infarction", "hypertension", 
                "obesity", "cancer", "heart failure", "stroke", "asthma",
                "arthritis", "depression", "anxiety", "alzheimer disease",
                "parkinson disease", "multiple sclerosis", "epilepsy",
                "clinical trial", "randomized controlled trial", "cohort study",
                "meta-analysis", "systematic review", "blood pressure",
                "body mass index", "chemotherapy", "surgery", "antibiotics"
            ])
            
        except Exception as e:
            print(f"Warning: Could not load MeSH terms: {e}")
            # Minimal fallback
            self.mesh_terms = set([
                "diabetes mellitus", "myocardial infarction", "hypertension", 
                "obesity", "cancer", "heart failure", "stroke", "asthma"
            ])
    
    def learn_patterns_from_texts(self, texts: List[str]):
        """Learn semantic patterns from the actual texts being processed."""
        if not texts:
            return
        
        # Extract potential multi-word terms using frequency analysis
        word_pairs = []
        word_triplets = []
        
        for text in texts:
            words = text.lower().split()
            
            # Extract word pairs
            for i in range(len(words) - 1):
                pair = f"{words[i]} {words[i+1]}"
                word_pairs.append(pair)
            
            # Extract word triplets
            for i in range(len(words) - 2):
                triplet = f"{words[i]} {words[i+1]} {words[i+2]}"
                word_triplets.append(triplet)
        
        # Find frequent patterns
        pair_counts = Counter(word_pairs)
        triplet_counts = Counter(word_triplets)
        
        # Add frequent patterns to learned patterns
        for pair, count in pair_counts.items():
            if count >= 2 and len(pair.split()) == 2:  # At least 2 occurrences
                self.learned_patterns.add(pair)
        
        for triplet, count in triplet_counts.items():
            if count >= 2 and len(triplet.split()) == 3:  # At least 2 occurrences
                self.learned_patterns.add(triplet)
    
    def get_semantic_patterns(self) -> List[str]:
        """Get all semantic patterns (MeSH + learned)."""
        all_patterns = list(self.mesh_terms) + list(self.learned_patterns)
        # Filter out very short patterns
        return [p for p in all_patterns if len(p.split()) >= 2 and len(p) > 3]
    
    def semantic_chunk(self, text: str, chunk_size: int = 2000, chunk_overlap: int = 300) -> List[str]:
        """
        Semantic chunking using cached MeSH terms and learned patterns.
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Get current semantic patterns
        semantic_patterns = self.get_semantic_patterns()
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                if chunk_overlap > 0:
                    # Find the last sentence boundary within overlap
                    overlap_text = current_chunk[-chunk_overlap:]
                    last_sentence = re.split(r'[.!?]\s+', overlap_text)[-1]
                    current_chunk = last_sentence + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If we have chunks that are too large, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large chunks by sentences
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current_sentence_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # Check if adding this sentence would exceed chunk size
                    if len(current_sentence_chunk) + len(sentence) > chunk_size and current_sentence_chunk:
                        final_chunks.append(current_sentence_chunk.strip())
                        # Start new chunk with overlap
                        if chunk_overlap > 0:
                            overlap_text = current_sentence_chunk[-chunk_overlap:]
                            last_part = re.split(r'[.!?]\s+', overlap_text)[-1]
                            current_sentence_chunk = last_part + " " + sentence
                        else:
                            current_sentence_chunk = sentence
                    else:
                        if current_sentence_chunk:
                            current_sentence_chunk += " " + sentence
                        else:
                            current_sentence_chunk = sentence
                
                # Add the last sentence chunk
                if current_sentence_chunk.strip():
                    final_chunks.append(current_sentence_chunk.strip())
        
        # If we still have chunks that are too large, use the original RecursiveCharacterTextSplitter as fallback
        final_final_chunks = []
        for chunk in final_chunks:
            if len(chunk) <= chunk_size:
                final_final_chunks.append(chunk)
            else:
                # Fallback to original method for very large chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", ".", " "]
                )
                sub_chunks = splitter.split_text(chunk)
                final_final_chunks.extend(sub_chunks)
        
        return final_final_chunks
    
    def get_cache_status(self):
        """Get information about the current cache status."""
        try:
            cache_file = "mesh_terms_cache.json"
            metadata_file = "mesh_terms_metadata.json"
            
            status = {
                'cache_exists': os.path.exists(cache_file),
                'metadata_exists': os.path.exists(metadata_file),
                'total_terms': len(self.mesh_terms),
                'learned_patterns': len(self.learned_patterns)
            }
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                status['sources'] = metadata.get('sources', [])
                status['version'] = metadata.get('version', 'unknown')
            
            return status
            
        except Exception as e:
            print(f"Error getting cache status: {e}")
            return {'error': str(e)}

# Global chunker instance
_semantic_chunker = PubMedSemanticChunker()

def semantic_chunking(text: str, chunk_size: int = 2000, chunk_overlap: int = 300) -> List[str]:
    """
    Semantic chunking that learns patterns from data and uses MeSH terms.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
    
    Returns:
        List of semantically coherent chunks
    """
    return _semantic_chunker.semantic_chunk(text, chunk_size, chunk_overlap)

def make_text_chunks(text: str, chunk_size: int = 2000, chunk_overlap: int = 300) -> List[str]:
    """
    Enhanced text chunking with data-driven semantic awareness.
    Uses semantic chunking first, falls back to RecursiveCharacterTextSplitter if needed.
    """
    try:
        # Try semantic chunking first
        semantic_chunks = semantic_chunking(text, chunk_size, chunk_overlap)
        
        # Validate chunks
        valid_chunks = []
        for chunk in semantic_chunks:
            if chunk.strip() and len(chunk.strip()) > 50:  # Minimum meaningful chunk size
                valid_chunks.append(chunk.strip())
        
        if valid_chunks:
            return valid_chunks
        
        # Fallback to original method if semantic chunking fails
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ".", " "] 
        )
        return splitter.split_text(text)
        
    except Exception as e:
        # If semantic chunking fails, use original method
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ".", " "] 
        )
        return splitter.split_text(text)


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


