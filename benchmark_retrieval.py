import argparse
import json
from typing import List, Tuple

import faiss
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from backend.embeddings import get_embedding_model, encode_texts


def semantic_chunk(text: str, max_len: int = 200) -> List[str]:
    """Chunk a long context into smaller pieces for embedding.

    Uses simple token-count (word) chunks with no overlap for speed.
    """
    words = text.split()
    chunks: List[str] = []
    for i in range(0, len(words), max_len):
        chunk = " ".join(words[i : i + max_len]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def load_bioasq(bioasq_file: str) -> list:
    with open(bioasq_file, "r", encoding="utf-8") as f:
        return json.load(f)["data"]


def build_corpus(bioasq_file: str) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """Build a retrieval corpus from all BioASQ contexts.

    Returns:
      - all_chunks: list of chunk texts
      - chunk_to_qas: parallel list mapping each chunk to tuples of (qa_id, answers)
    """
    data = load_bioasq(bioasq_file)

    all_chunks: List[str] = []
    chunk_to_qas: List[Tuple[str, List[str]]] = []

    for para_group in data:
        for para in para_group.get("paragraphs", []):
            context: str = para.get("context", "")
            qas: list = para.get("qas", [])
            answers_for_para: List[Tuple[str, List[str]]] = []
            for qa in qas:
                qa_id = qa.get("id", "")
                gold_answers = [str(a.get("text", "")).lower() for a in qa.get("answers", [])]
                answers_for_para.append((qa_id, gold_answers))

            chunks = semantic_chunk(context)
            for chunk in chunks:
                all_chunks.append(chunk)
                # Associate this chunk with all QA answer sets in the same paragraph
                for qa_id, gold in answers_for_para:
                    chunk_to_qas.append((qa_id, gold))

    return all_chunks, chunk_to_qas


def build_faiss_index(chunks: List[str], model_name: str):
    model = get_embedding_model(model_name)
    embeddings = encode_texts(chunks, model)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings, model


def evaluate(bioasq_file: str, model_name: str, top_k: int = 5) -> Tuple[float, float]:
    # Build corpus and index
    all_chunks, _ = build_corpus(bioasq_file)
    index, _, model = build_faiss_index(all_chunks, model_name)

    data = load_bioasq(bioasq_file)
    recall_hits: List[int] = []
    ndcgs: List[float] = []

    for para_group in tqdm(data, desc=f"Evaluating ({model_name})"):
        for para in para_group.get("paragraphs", []):
            for qa in para.get("qas", []):
                question: str = qa.get("question", "")
                gold_answers = [str(a.get("text", "")).lower() for a in qa.get("answers", [])]
                if not question or not gold_answers:
                    continue

                q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
                D, I = index.search(q_emb, top_k)
                retrieved_chunks = [all_chunks[idx] for idx in I[0]]

                # Recall@k: whether any gold answer substring appears in any retrieved chunk
                hit = any(
                    any(ans in chunk.lower() for ans in gold_answers)
                    for chunk in retrieved_chunks
                )
                recall_hits.append(1 if hit else 0)

                # NDCG@k: binary relevance by substring presence
                relevance = [0] * top_k
                for i, chunk in enumerate(retrieved_chunks):
                    if any(ans in chunk.lower() for ans in gold_answers):
                        relevance[i] = 1
                ndcgs.append(ndcg_score([relevance], [list(D[0])], k=top_k))

    recall_at_k = float(np.mean(recall_hits)) if recall_hits else 0.0
    ndcg_at_k = float(np.mean(ndcgs)) if ndcgs else 0.0
    return recall_at_k, ndcg_at_k


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval on BioASQ with different embedding models.")
    parser.add_argument(
        "--bioasq",
        required=True,
        help="Path to BioASQ JSON file (e.g., BioASQ-train-factoid-6b-full-annotated.json)",
    )
    parser.add_argument(
        "--model",
        choices=[
            "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "jinaai/jina-embeddings-v2-base-en",
            "gemini-embeddings-001",
        ],
        default="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
        help="Embedding model to evaluate",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Top-k for retrieval metrics")

    args = parser.parse_args()

    recall_at_k, ndcg_at_k = evaluate(args.bioasq, args.model, top_k=args.top_k)
    print(f"Model: {args.model}")
    print(f"Recall@{args.top_k}: {recall_at_k:.3f}")
    print(f"NDCG@{args.top_k}: {ndcg_at_k:.3f}")


if __name__ == "__main__":
    main()


