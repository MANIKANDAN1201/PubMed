
"""
summary_cluster.py

Enhanced PubMed article summarization system using abstract embeddings as knowledge base.
- Uses abstract embeddings for semantic similarity and relevance scoring
- Generates comprehensive summaries based on embedding chunks
- Provides both individual article summaries and cluster-based summaries
- Ensures summaries are relevant to the user's specific query
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from improved_vector_store import ImprovedVectorStore
from embeddings import TextEmbedder
import re

# Conditional import to avoid dependency issues
try:
    from pubmed_fetcher import PubMedArticle, fetch_pubmed_articles
    HAS_PUBMED_FETCHER = True
except ImportError:
    # Create a minimal PubMedArticle class for testing
    from dataclasses import dataclass
    from typing import List, Optional
    
    @dataclass
    class PubMedArticle:
        pmid: str
        title: str
        abstract: str
        url: str
        journal: Optional[str] = None
        year: Optional[str] = None
        authors: Optional[List[str]] = None
        doi: Optional[str] = None
        is_free: bool = False
        full_text: Optional[str] = None
        full_text_link: Optional[str] = None
        free_source: Optional[str] = None
    
    def fetch_pubmed_articles(*args, **kwargs):
        return []
    
    HAS_PUBMED_FETCHER = False

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ArticleSummary:
    """Container for article summary with relevance information."""
    article: PubMedArticle
    summary: str
    relevance_score: float
    key_points: List[str]
    query_terms_found: List[str]


class AbstractEmbeddingSummarizer:
    """Main class for abstract-based summarization using embeddings as knowledge base."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.text_embedder = TextEmbedder(embedding_model)
        self.vector_store = ImprovedVectorStore()
        
    def prepare_abstract_texts(self, articles: List[PubMedArticle]) -> List[str]:
        """Prepare abstract texts for embedding and summarization."""
        texts = []
        for article in articles:
            # Combine title and abstract for better context
            text = f"{article.title}\n{article.abstract}"
            if getattr(article, 'is_free', False) and getattr(article, 'full_text', None):
                # Add full text if available for more comprehensive summaries
                text += f"\n{article.full_text[:1000]}"  # Limit full text length
            texts.append(text)
        return texts
    
    def calculate_relevance_scores(self, articles: List[PubMedArticle], query: str) -> List[float]:
        """Calculate relevance scores using embedding similarity."""
        query_embedding = self.text_embedder.encode([query])[0]
        
        relevance_scores = []
        for article in articles:
            # Create text representation
            article_text = f"{article.title}\n{article.abstract}"
            article_embedding = self.text_embedder.encode([article_text])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, article_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(article_embedding)
            )
            relevance_scores.append(float(similarity))
        
        return relevance_scores
    
    def create_knowledge_chunks(self, articles: List[PubMedArticle], query: str) -> List[Dict]:
        """Create knowledge chunks from articles based on query relevance."""
        # Calculate relevance scores
        relevance_scores = self.calculate_relevance_scores(articles, query)
        
        # Create knowledge chunks with metadata
        knowledge_chunks = []
        for i, article in enumerate(articles):
            chunk = {
                'article': article,
                'text': f"{article.title}\n{article.abstract}",
                'relevance_score': relevance_scores[i],
                'pmid': article.pmid,
                'chunk_type': 'abstract'
            }
            
            # Add full text chunk if available
            if getattr(article, 'is_free', False) and getattr(article, 'full_text', None):
                full_text_chunk = {
                    'article': article,
                    'text': article.full_text[:2000],  # Limit length
                    'relevance_score': relevance_scores[i] * 0.8,  # Slightly lower weight
                    'pmid': article.pmid,
                    'chunk_type': 'full_text'
                }
                knowledge_chunks.append(full_text_chunk)
            
            knowledge_chunks.append(chunk)
        
        # Sort by relevance score
        knowledge_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        return knowledge_chunks
    
    def extract_relevant_sentences(self, text: str, query: str, max_sentences: int = 8) -> List[str]:
        """Extract the most relevant sentences for comprehensive summaries."""
        query_terms = query.lower().split()
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        
        # Score each sentence based on query term frequency and medical relevance
        sentence_scores = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            score = 0
            
            # Count query terms in sentence
            for term in query_terms:
                score += sentence_lower.count(term) * 3
            
            # Bonus for sentences that start with query terms
            if any(sentence_lower.startswith(term) for term in query_terms):
                score += 5
            
            # Bonus for sentences with medical information
            medical_keywords = ['symptom', 'sign', 'treat', 'cause', 'risk', 'effect', 'result', 
                              'study', 'research', 'clinical', 'patient', 'diagnosis', 'therapy']
            for keyword in medical_keywords:
                if keyword in sentence_lower:
                    score += 2
            
            # Bonus for sentences with specific medical details
            specific_terms = ['pain', 'disease', 'condition', 'treatment', 'medication', 'test']
            for term in specific_terms:
                if term in sentence_lower:
                    score += 1
            
            if score > 0:
                sentence_scores.append((sentence, score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        for sentence, _ in sentence_scores[:max_sentences]:
            relevant_sentences.append(sentence)
        
        return relevant_sentences
    
    def generate_comprehensive_summary(self, knowledge_chunks: List[Dict], query: str) -> str:
        """Generate comprehensive summary from knowledge chunks."""
        if not knowledge_chunks:
            return f"Based on the available information about {query}, consult healthcare professionals for personalized medical advice."
        
        # Extract relevant content from top chunks
        all_relevant_sentences = []
        
        for chunk in knowledge_chunks[:5]:  # Use top 5 most relevant chunks
            relevant_sentences = self.extract_relevant_sentences(chunk['text'], query)
            all_relevant_sentences.extend(relevant_sentences)
        
        if not all_relevant_sentences:
            return f"Based on the available information about {query}, consult healthcare professionals for personalized medical advice."
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for sentence in all_relevant_sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        # Create structured summary
        summary_parts = []
        
        # Group sentences by type
        symptoms_sentences = []
        treatment_sentences = []
        cause_sentences = []
        general_sentences = []
        
        for sentence in unique_sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in ['symptom', 'sign', 'indication', 'warning', 'feel', 'experience']):
                symptoms_sentences.append(sentence)
            elif any(word in sentence_lower for word in ['treat', 'therapy', 'medication', 'management', 'control', 'prevent']):
                treatment_sentences.append(sentence)
            elif any(word in sentence_lower for word in ['cause', 'risk', 'factor', 'trigger', 'lead to']):
                cause_sentences.append(sentence)
            else:
                general_sentences.append(sentence)
        
        # Build comprehensive summary
        if symptoms_sentences:
            summary_parts.append("**Symptoms and Signs:** " + ". ".join(symptoms_sentences[:4]) + ".")
        
        if cause_sentences:
            summary_parts.append("**Causes and Risk Factors:** " + ". ".join(cause_sentences[:3]) + ".")
        
        if treatment_sentences:
            summary_parts.append("**Treatment and Management:** " + ". ".join(treatment_sentences[:3]) + ".")
        
        if general_sentences and not summary_parts:
            # If no specific categories, use general info
            summary_parts.append(". ".join(general_sentences[:4]) + ".")
        
        # Combine all parts
        final_summary = " ".join(summary_parts)
        
        # Ensure reasonable length
        if len(final_summary) < 200:
            # Add more content if summary is too short
            remaining_sentences = unique_sentences[len(summary_parts):]
            if remaining_sentences:
                final_summary += " " + ". ".join(remaining_sentences[:3]) + "."
        
        # Clean up and ensure proper punctuation
        final_summary = re.sub(r'\s+', ' ', final_summary).strip()
        if not final_summary.endswith(('.', '!', '?')):
            final_summary += '.'
        
        return final_summary
    
    def summarize_all_articles(self, articles: List[PubMedArticle], query: str) -> List[ArticleSummary]:
        """Summarize all articles using embedding-based knowledge chunks."""
        if not articles:
            return []
        
        # Create knowledge chunks
        knowledge_chunks = self.create_knowledge_chunks(articles, query)
        
        # Generate comprehensive summary
        comprehensive_summary = self.generate_comprehensive_summary(knowledge_chunks, query)
        
        # Create individual article summaries
        summaries = []
        for chunk in knowledge_chunks:
            if chunk['chunk_type'] == 'abstract':  # Only create one summary per article
                article = chunk['article']
                
                # Generate article-specific summary
                article_summary = self.generate_comprehensive_summary([chunk], query)
                
                # Extract key points
                key_points = self.extract_relevant_sentences(chunk['text'], query, max_sentences=3)
                
                # Find query terms
                query_terms = query.lower().split()
                found_terms = [term for term in query_terms if term in chunk['text'].lower()]
                
                summary = ArticleSummary(
                    article=article,
                    summary=article_summary,
                    relevance_score=chunk['relevance_score'],
                    key_points=key_points,
                    query_terms_found=found_terms
                )
                summaries.append(summary)
        
        return summaries
    
    def cluster_and_summarize(self, articles: List[PubMedArticle], query: str, n_clusters: int = 3) -> Dict:
        """Cluster articles and provide cluster-based summaries."""
        if len(articles) < n_clusters:
            n_clusters = len(articles)
        
        # Create knowledge chunks
        knowledge_chunks = self.create_knowledge_chunks(articles, query)
        
        # Extract embeddings for clustering
        chunk_texts = [chunk['text'] for chunk in knowledge_chunks]
        chunk_embeddings = self.text_embedder.encode(chunk_texts)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(chunk_embeddings)
        
        # Group chunks by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(knowledge_chunks[i])
        
        # Generate cluster summaries
        cluster_summaries = {}
        for cluster_id, cluster_chunks in clusters.items():
            # Sort chunks within cluster by relevance
            cluster_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Generate summary for this cluster
            cluster_summary = self.generate_comprehensive_summary(cluster_chunks, query)
            
            cluster_summaries[cluster_id] = {
                'summary': cluster_summary,
                'articles': [chunk['article'] for chunk in cluster_chunks],
                'relevance_score': np.mean([chunk['relevance_score'] for chunk in cluster_chunks])
            }
        
        return cluster_summaries
    
    def get_top_summaries(self, articles: List[PubMedArticle], query: str, top_n: int = 5) -> List[ArticleSummary]:
        """Get top N most relevant article summaries."""
        summaries = self.summarize_all_articles(articles, query)
        
        # Sort by relevance score and return top N
        summaries.sort(key=lambda x: x.relevance_score, reverse=True)
        return summaries[:top_n]


# Compatibility functions for existing code
def prepare_texts_for_embedding(articles: List[PubMedArticle]) -> List[str]:
    """Prepare texts for embedding - maintains compatibility with existing code."""
    texts = []
    for art in articles:
        if getattr(art, 'is_free', False) and getattr(art, 'full_text', None):
            # Use full text + abstract for free articles
            text = f"{art.title}\n{art.abstract}\n{art.full_text}"
        else:
            text = f"{art.title}\n{art.abstract}"
        texts.append(text)
    return texts


def summarize_top_articles(sorted_results, query, top_n=5):
    """Summarize top N most relevant articles - maintains compatibility with existing code."""
    top_n = min(top_n, len(sorted_results))
    sorted_by_score = sorted(sorted_results, key=lambda x: x["score"], reverse=True)
    
    # Extract articles
    articles = [result['art'] for result in sorted_by_score[:top_n]]
    
    # Create summarizer and generate summaries
    summarizer = AbstractEmbeddingSummarizer()
    summaries = summarizer.get_top_summaries(articles, query, top_n)
    
    if summaries:
        # Return the most relevant summary
        return summaries[0].summary
    else:
        return f"Based on the available information about {query}, consult healthcare professionals for personalized medical advice."


def summarize_free_full_texts(sorted_results, query):
    """Summarize free full-text articles - maintains compatibility with existing code."""
    free_articles = []
    for result in sorted_results:
        art = result["art"]
        if getattr(art, 'is_free', False) and getattr(art, 'full_text', None):
            free_articles.append(art)
    
    if free_articles:
        summarizer = AbstractEmbeddingSummarizer()
        summaries = summarizer.get_top_summaries(free_articles, query, len(free_articles))
        
        if summaries:
            return summaries[0].summary
    
    return None


if __name__ == "__main__":
    print("Abstract Embedding Summarization System")
    print("Use debug_summary.py to test the functionality")
