from __future__ import annotations
from typing import List, Tuple, Any, Optional

# Import LangChain FlashRank reranker class
try:
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
    from langchain_core.documents import Document
except ImportError:
    try:
        # Fallback to old import path
        from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
        from langchain_core.documents import Document
    except ImportError:
        FlashrankRerank = None
        Document = None


def _prepare_docs(articles: List[Any], keep_indices: List[int], local_indices: List[int]) -> List[Document]:
    docs: List[Document] = []
    if Document is None:
        return docs

    for idx in local_indices:
        if idx < 0 or idx >= len(keep_indices):
            continue
        art = articles[keep_indices[idx]]
        title = getattr(art, "title", "") or ""
        abstract = getattr(art, "abstract", "") or ""
        txt = (f"{title}\n\n{abstract}".strip()) or title or abstract or ""
        docs.append(Document(page_content=txt, metadata={"index": idx}))
    return docs


def _extract_item_id(doc: Any) -> Optional[int]:
    try:
        return int(doc.metadata.get("index"))  # for Document
    except Exception:
        return None


def flashrank_rerank(
    query: str,
    articles: List[Any],
    keep_indices: List[int],
    scores: List[float],
    indices: List[int],
    result_metadata: List[Any],
    model: str = "ms-marco-TinyBERT-L-2-v2",
) -> Tuple[List[float], List[int], List[Any]]:
    if FlashrankRerank is None or Document is None:
        return scores, indices, result_metadata

    docs = _prepare_docs(articles, keep_indices, list(indices))
    if not docs:
        return scores, indices, result_metadata

    try:
        reranker = FlashrankRerank(model=model, top_n=len(docs))
    except Exception as e:
        print(f"FlashRank initialization failed: {e}")
        return scores, indices, result_metadata

    try:
        reranked_docs = reranker.compress_documents(docs, query=query)
    except Exception as e:
        print(f"FlashRank rerank failed: {e}")
        return scores, indices, result_metadata

    order_local: List[int] = []
    for doc in reranked_docs:
        idx = _extract_item_id(doc)
        if idx is not None:
            order_local.append(idx)

    if not order_local:
        return scores, indices, result_metadata

    idx_to_tuple = {
        int(i): (float(s), int(i), m)
        for s, i, m in zip(scores, indices, result_metadata)
    }

    reordered = [idx_to_tuple[i] for i in order_local if i in idx_to_tuple]
    if not reordered:
        return scores, indices, result_metadata

    new_scores, new_indices, new_meta = zip(*reordered)
    return list(new_scores), list(new_indices), list(new_meta)

