import json
import numpy as np
from tqdm import tqdm
import faiss
from sklearn.metrics import ndcg_score
import google.generativeai as genai
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import ReadTimeout, ConnectionError

# -----------------------------
# Configure Gemini API key
# -----------------------------
genai.configure(api_key="AIzaSyCyt08zIKOV75kct_w3dbaAQc00IG0UwGg")  # Replace with your key

EMBED_MODEL = "models/embedding-001"

# -----------------------------
# Chunk text into smaller parts
# -----------------------------
def semantic_chunk(text, max_len=200):
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len) if words[i:i+max_len]]

# -----------------------------
# Load BioASQ dataset
# -----------------------------
def load_dataset(bioasq_file):
    with open(bioasq_file, "r", encoding="utf-8") as f:
        return json.load(f)["data"]

# -----------------------------
# Safe batch embedding with retry
# -----------------------------
def get_gemini_embedding_batch(texts, retries=3):
    for attempt in range(retries):
        try:
            response = genai.embed_content(model=EMBED_MODEL, content=texts)
            if isinstance(response, list):
                return [np.array(item["embedding"], dtype=np.float32) for item in response]
            elif isinstance(response, dict) and "embedding" in response:
                return [np.array(response["embedding"], dtype=np.float32)]
            raise TypeError(f"Unexpected response: {response}")
        except (ReadTimeout, ConnectionError) as e:
            wait_time = 2 ** attempt
            print(f"⚠ Retry in {wait_time}s due to {type(e).__name__} (Attempt {attempt+1}/{retries})")
            time.sleep(wait_time)
    raise TimeoutError("❌ Batch failed after retries.")

# -----------------------------
# Build retrieval corpus
# -----------------------------
def build_corpus(data):
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

# -----------------------------
# Build FAISS index (with optional parallelism and caching)
# -----------------------------
def build_faiss_index(chunks, batch_size=8, max_workers=2, cache_file="chunk_embeddings.npy"):
    if os.path.exists(cache_file):
        print(f"💾 Loading cached embeddings from {cache_file}")
        embeddings = np.load(cache_file)
    else:
        embeddings = []

        # Sequential fallback if max_workers <= 1
        if max_workers <= 1:
            for i in tqdm(range(0, len(chunks), batch_size), desc="Encoding chunks (sequential)"):
                batch = chunks[i:i+batch_size]
                embeddings.extend(get_gemini_embedding_batch(batch))
        else:
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    futures.append(executor.submit(get_gemini_embedding_batch, batch))
                for f in tqdm(as_completed(futures), total=len(futures), desc="Encoding chunks (parallel)"):
                    embeddings.extend(f.result())

        embeddings = np.vstack(embeddings)
        np.save(cache_file, embeddings)
        print(f"💾 Saved embeddings to {cache_file}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings

# -----------------------------
# Embed questions (with caching)
# -----------------------------
def embed_questions(questions, batch_size=4, cache_file="question_embeddings.npy"):
    if os.path.exists(cache_file):
        print(f"💾 Loading cached question embeddings from {cache_file}")
        return np.load(cache_file)

    q_embeddings = []
    for i in tqdm(range(0, len(questions), batch_size), desc="Encoding questions"):
        batch = questions[i:i+batch_size]
        q_embeddings.extend(get_gemini_embedding_batch(batch))

    q_embeddings = np.vstack(q_embeddings)
    np.save(cache_file, q_embeddings)
    print(f"💾 Saved question embeddings to {cache_file}")
    return q_embeddings

# -----------------------------
# Evaluate model
# -----------------------------
def evaluate_model(data, top_k=5, chunk_batch_size=8, question_batch_size=4, max_workers=2):
    print(f"\n🔹 Evaluating model: {EMBED_MODEL}")
    all_chunks, _ = build_corpus(data)
    index, _ = build_faiss_index(all_chunks, batch_size=chunk_batch_size, max_workers=max_workers)

    questions = []
    gold_answers_list = []
    for para_group in data:
        for para in para_group["paragraphs"]:
            for qa in para["qas"]:
                questions.append(qa["question"])
                gold_answers_list.append([a["text"].lower() for a in qa["answers"]])

    q_embeddings = embed_questions(questions, batch_size=question_batch_size)

    recall_hits, ndcg_vals = [], []
    for q_emb, gold_answers in zip(q_embeddings, gold_answers_list):
        q_emb = q_emb.reshape(1, -1)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, top_k)
        retrieved_chunks = [all_chunks[idx] for idx in I[0]]

        hit = any(any(ans in chunk.lower() for ans in gold_answers) for chunk in retrieved_chunks)
        recall_hits.append(1 if hit else 0)

        relevance = [1 if any(ans in chunk.lower() for ans in gold_answers) else 0 for chunk in retrieved_chunks]
        ndcg_vals.append(ndcg_score([relevance], [list(D[0])], k=top_k))

    return {
        "model": EMBED_MODEL,
        f"Recall@{top_k}": float(np.mean(recall_hits)),
        f"NDCG@{top_k}": float(np.mean(ndcg_vals))
    }

# -----------------------------
# Main
# -----------------------------
def evaluate(bioasq_file, top_k=5, chunk_batch_size=8, question_batch_size=4, max_workers=2):
    data = load_dataset(bioasq_file)
    return evaluate_model(data, top_k=top_k,
                          chunk_batch_size=chunk_batch_size,
                          question_batch_size=question_batch_size,
                          max_workers=max_workers)

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    result = evaluate("BioASQ-train-factoid-6b-full-annotated.json",
                      top_k=5,
                      chunk_batch_size=4,    # smaller batch for stability
                      question_batch_size=2, # smaller batch for stability
                      max_workers=1)         # sequential for reliability
    print("\n📊 Final Results:")
    print(result)
