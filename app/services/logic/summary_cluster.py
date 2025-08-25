import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .improved_vector_store import ImprovedVectorStore
from .embeddings import TextEmbedder
import re

try:
    from .pubmed_fetcher import PubMedArticle, fetch_pubmed_articles
    HAS_PUBMED_FETCHER = True
except ImportError:
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


@dataclass
class ArticleSummary:
    article: PubMedArticle
    summary: str
    relevance_score: float
    key_points: List[str]
    query_terms_found: List[str]


class AbstractEmbeddingSummarizer:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.text_embedder = TextEmbedder(embedding_model)
        self.vector_store = ImprovedVectorStore()
        
    def calculate_relevance_scores(self, articles: List[PubMedArticle], query: str) -> List[float]:
        query_embedding = self.text_embedder.encode([query])[0]
        relevance_scores = []
        for article in articles:
            article_text = f"{article.title}\n{article.abstract}"
            article_embedding = self.text_embedder.encode([article_text])[0]
            similarity = np.dot(query_embedding, article_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(article_embedding)
            )
            relevance_scores.append(float(similarity))
        return relevance_scores
    
    def create_knowledge_chunks(self, articles: List[PubMedArticle], query: str) -> List[Dict]:
        relevance_scores = self.calculate_relevance_scores(articles, query)
        knowledge_chunks = []
        for i, article in enumerate(articles):
            chunk = {
                'article': article,
                'text': f"{article.title}\n{article.abstract}",
                'relevance_score': relevance_scores[i],
                'pmid': article.pmid,
                'chunk_type': 'abstract'
            }
            if getattr(article, 'is_free', False) and getattr(article, 'full_text', None):
                full_text_chunk = {
                    'article': article,
                    'text': article.full_text[:2000],
                    'relevance_score': relevance_scores[i] * 0.8,
                    'pmid': article.pmid,
                    'chunk_type': 'full_text'
                }
                knowledge_chunks.append(full_text_chunk)
            knowledge_chunks.append(chunk)
        knowledge_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        return knowledge_chunks
    
    def extract_relevant_sentences(self, text: str, query: str, max_sentences: int = 8) -> List[str]:
        query_terms = query.lower().split()
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        sentence_scores = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            sentence_lower = sentence.lower()
            score = 0
            for term in query_terms:
                score += sentence_lower.count(term) * 3
            if any(sentence_lower.startswith(term) for term in query_terms):
                score += 5
            medical_keywords = ['symptom', 'sign', 'treat', 'cause', 'risk', 'effect', 'result', 'study', 'research', 'clinical', 'patient', 'diagnosis', 'therapy']
            for keyword in medical_keywords:
                if keyword in sentence_lower:
                    score += 2
            specific_terms = ['pain', 'disease', 'condition', 'treatment', 'medication', 'test']
            for term in specific_terms:
                if term in sentence_lower:
                    score += 1
            if score > 0:
                sentence_scores.append((sentence, score))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        for sentence, _ in sentence_scores[:max_sentences]:
            relevant_sentences.append(sentence)
        return relevant_sentences
    
    def generate_comprehensive_summary(self, knowledge_chunks: List[Dict], query: str) -> str:
        if not knowledge_chunks:
            return f"Based on the available information about {query}, consult healthcare professionals for personalized medical advice."
        all_relevant_sentences = []
        for chunk in knowledge_chunks[:5]:
            relevant_sentences = self.extract_relevant_sentences(chunk['text'], query)
            all_relevant_sentences.extend(relevant_sentences)
        if not all_relevant_sentences:
            return f"Based on the available information about {query}, consult healthcare professionals for personalized medical advice."
        seen = set()
        unique_sentences = []
        for sentence in all_relevant_sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        summary_parts = []
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
        if symptoms_sentences:
            summary_parts.append("**Symptoms and Signs:** " + ". ".join(symptoms_sentences[:4]) + ".")
        if cause_sentences:
            summary_parts.append("**Causes and Risk Factors:** " + ". ".join(cause_sentences[:3]) + ".")
        if treatment_sentences:
            summary_parts.append("**Treatment and Management:** " + ". ".join(treatment_sentences[:3]) + ".")
        if general_sentences and not summary_parts:
            summary_parts.append(". ".join(general_sentences[:4]) + ".")
        final_summary = " ".join(summary_parts)
        if len(final_summary) < 200:
            remaining_sentences = unique_sentences[len(summary_parts):]
            if remaining_sentences:
                final_summary += " " + ". ".join(remaining_sentences[:3]) + "."
        final_summary = re.sub(r'\s+', ' ', final_summary).strip()
        if not final_summary.endswith(('.', '!', '?')):
            final_summary += '.'
        return final_summary

    def summarize_all_articles(self, articles: List[PubMedArticle], query: str) -> List[ArticleSummary]:
        if not articles:
            return []
        knowledge_chunks = self.create_knowledge_chunks(articles, query)
        summaries: List[ArticleSummary] = []
        for chunk in knowledge_chunks:
            if chunk.get('chunk_type') == 'abstract':
                article = chunk['article']
                article_summary = self.generate_comprehensive_summary([chunk], query)
                key_points = self.extract_relevant_sentences(chunk['text'], query, max_sentences=3)
                query_terms = query.lower().split()
                found_terms = [term for term in query_terms if term in chunk['text'].lower()]
                summaries.append(ArticleSummary(
                    article=article,
                    summary=article_summary,
                    relevance_score=chunk['relevance_score'],
                    key_points=key_points,
                    query_terms_found=found_terms,
                ))
        return summaries

    def get_top_summaries(self, articles: List[PubMedArticle], query: str, top_n: int = 5) -> List[ArticleSummary]:
        summaries = self.summarize_all_articles(articles, query)
        summaries.sort(key=lambda x: x.relevance_score, reverse=True)
        return summaries[:top_n]

    def cluster_and_summarize(self, articles: List[PubMedArticle], query: str, n_clusters: int = 3) -> Dict:
        if len(articles) < n_clusters:
            n_clusters = len(articles) if len(articles) > 0 else 1
        knowledge_chunks = self.create_knowledge_chunks(articles, query)
        chunk_texts = [chunk['text'] for chunk in knowledge_chunks]
        if not chunk_texts:
            return {}
        chunk_embeddings = self.text_embedder.encode(chunk_texts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(chunk_embeddings)
        clusters: Dict[int, List[Dict]] = {}
        for i, label in enumerate(cluster_labels):
            clusters.setdefault(int(label), []).append(knowledge_chunks[i])
        cluster_summaries: Dict[int, Dict] = {}
        for cluster_id, cluster_chunks in clusters.items():
            cluster_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
            cluster_summary = self.generate_comprehensive_summary(cluster_chunks, query)
            cluster_summaries[int(cluster_id)] = {
                'summary': cluster_summary,
                'articles': [chunk['article'] for chunk in cluster_chunks],
                'relevance_score': float(np.mean([chunk['relevance_score'] for chunk in cluster_chunks])) if cluster_chunks else 0.0,
            }
        return cluster_summaries


def prepare_texts_for_embedding(articles: List[PubMedArticle]) -> List[str]:
    texts = []
    for art in articles:
        if getattr(art, 'is_free', False) and getattr(art, 'full_text', None):
            text = f"{art.title}\n{art.abstract}\n{art.full_text}"
        else:
            text = f"{art.title}\n{art.abstract}"
        texts.append(text)
    return texts


