import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics import ndcg_score


def semantic_chunk(text, max_len=200):
    """Chunk context into smaller overlapping pieces for embeddings."""
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len) if words[i:i+max_len]]


def load_dataset(bioasq_file):
    """Load dataset in SQuAD-style JSON format."""
    with open(bioasq_file, "r", encoding="utf-8") as f:
        return json.load(f)["data"]


def build_corpus(data):
    """Build a retrieval corpus from all contexts."""
    all_chunks, chunk_to_qid = [], []
    for para_group in data:
        for para in para_group["paragraphs"]:
            context = para["context"]
            qas = para["qas"]

            chunks = semantic_chunk(context)
            for chunk in chunks:
                all_chunks.append(chunk)
                for qa in qas:
                    chunk_to_qid.append((qa["id"], qa["answers"]))
    return all_chunks, chunk_to_qid


def build_faiss_index(model, chunks):
    """Build FAISS index of embeddings for all chunks."""
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)  # cosine similarity
    index.add(embeddings)

    return index, embeddings


def evaluate_model(model_name, data, top_k=5):
    """Evaluate one embedding model."""
    print(f"\n🔹 Evaluating model: {model_name}")
    model = SentenceTransformer(model_name)

    # Step 1: build corpus & FAISS
    all_chunks, _ = build_corpus(data)
    index, _ = build_faiss_index(model, all_chunks)

    # Step 2: evaluation loop
    recall_hits, ndcg_vals = [], []
    for para_group in tqdm(data, desc=f"Evaluating {model_name}"):
        for para in para_group["paragraphs"]:
            for qa in para["qas"]:
                question = qa["question"]
                gold_answers = [a["text"].lower() for a in qa["answers"]]

                # Encode query
                q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)

                # Retrieve top-k
                D, I = index.search(q_emb, top_k)
                retrieved_chunks = [all_chunks[idx] for idx in I[0]]

                # Recall@k
                hit = any(any(ans in chunk.lower() for ans in gold_answers) for chunk in retrieved_chunks)
                recall_hits.append(1 if hit else 0)

                # NDCG@k
                relevance = [0] * top_k
                for i, chunk in enumerate(retrieved_chunks):
                    if any(ans in chunk.lower() for ans in gold_answers):
                        relevance[i] = 1
                ndcg_vals.append(ndcg_score([relevance], [list(D[0])], k=top_k))

    return {
        "model": model_name,
        f"Recall@{top_k}": float(np.mean(recall_hits)),
        f"NDCG@{top_k}": float(np.mean(ndcg_vals))
    }


def evaluate_multiple_models(bioasq_file, model_names, top_k=5):
    """Run evaluation across multiple embedding models."""
    data = load_dataset(bioasq_file)

    results = []
    for model_name in model_names:
        res = evaluate_model(model_name, data, top_k=top_k)
        results.append(res)

    return results


if __name__ == "__main__":
    models = [
        "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "jinaai/jina-embeddings-v2-base-en"
    ]
    results = evaluate_multiple_models("BioASQ-train-factoid-6b-full-annotated.json", models, top_k=5)
    print("\n Final Results:")
    for r in results:
        print(r)
