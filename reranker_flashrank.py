from __future__ import annotations
from typing import List, Tuple, Any, Optional

# Be resilient to different flashrank versions
try:
    import flashrank  # type: ignore
    from flashrank import Ranker, RerankRequest  # type: ignore
except Exception:  # flashrank not installed or import shape changed
    flashrank = None
    Ranker = None
    RerankRequest = None


def _prepare_docs(articles: List[Any], keep_indices: List[int], local_indices: List[int]) -> List[dict]:
    """
    Prepare documents for FlashRank reranker.
    Use both 'text' and 'content' keys to satisfy older/newer versions.
    """
    docs = []
    for idx in local_indices:
        if idx < 0 or idx >= len(keep_indices):
            continue
        art = articles[keep_indices[idx]]
        title = getattr(art, "title", "") or ""
        abstract = getattr(art, "abstract", "") or ""
        txt = (f"{title}\n\n{abstract}".strip()) or title or abstract or ""
        docs.append({
            "id": str(idx),
            "text": txt,
            "content": txt, 
        })
    return docs


def _extract_item_id(item: Any) -> Optional[int]:
    """
    Extract the local index id from a flashrank result item across versions.
    """
    # object with attribute 'document'
    try:
        doc = getattr(item, "document", None)
        if doc is not None:
            # doc can be dict-like or an object with 'id'
            if isinstance(doc, dict):
                return int(doc.get("id"))
            _id = getattr(doc, "id", None)
            if _id is not None:
                return int(_id)
    except Exception:
        pass

    # dict shapes
    try:
        if isinstance(item, dict):
            if "document" in item and isinstance(item["document"], dict) and "id" in item["document"]:
                return int(item["document"]["id"])
            if "doc" in item and isinstance(item["doc"], dict) and "id" in item["doc"]:
                return int(item["doc"]["id"])
            if "id" in item:
                return int(item["id"])
    except Exception:
        pass

    # last resort
    return None


def _call_ranker_any(ranker: Any, query: str, docs: List[dict], k: int):
    """
    Try all known call patterns for different flashrank versions.
    Returns (ranked_list, debug_message) or (None, debug_message).
    """
    errs = []

    # 1) Newer style: explicit request object (documents=)
    if RerankRequest is not None:
        try:
            req = RerankRequest(query=query, documents=docs)  # some versions accept 'documents'
            ranked = ranker.rerank(req, top_k=k)
            return ranked, "used RerankRequest(documents=...)"
        except Exception as e:
            errs.append(f"RerankRequest(documents=...): {e}")

        try:
            req = RerankRequest(query=query, docs=docs)  # some versions accept 'docs'
            ranked = ranker.rerank(req, top_k=k)
            return ranked, "used RerankRequest(docs=...)"
        except Exception as e:
            errs.append(f"RerankRequest(docs=...): {e}")

        try:
            req = RerankRequest(query, docs)  # positional
            ranked = ranker.rerank(req, top_k=k)
            return ranked, "used RerankRequest(positional)"
        except Exception as e:
            errs.append(f"RerankRequest(positional): {e}")

    # 2) Some versions accept calling rerank directly with kwargs (documents/docs)
    try:
        ranked = ranker.rerank(query=query, documents=docs, top_k=k)
        return ranked, "used rerank(query=..., documents=..., top_k=...)"
    except Exception as e:
        errs.append(f"rerank(query, documents): {e}")

    try:
        ranked = ranker.rerank(query=query, docs=docs, top_k=k)
        return ranked, "used rerank(query=..., docs=..., top_k=...)"
    except Exception as e:
        errs.append(f"rerank(query, docs): {e}")

    # 3) Pure positional call
    try:
        ranked = ranker.rerank(query, docs, top_k=k)
        return ranked, "used rerank(positional)"
    except Exception as e:
        errs.append(f"rerank(positional): {e}")

    # 4) Try with 'content' only (strip 'text')
    docs_content = [{"id": d["id"], "content": d.get("content") or d.get("text", "")} for d in docs]

    if RerankRequest is not None:
        try:
            req = RerankRequest(query=query, documents=docs_content)
            ranked = ranker.rerank(req, top_k=k)
            return ranked, "used RerankRequest(documents=...) with 'content' only"
        except Exception as e:
            errs.append(f"RerankRequest(documents, content-only): {e}")

        try:
            req = RerankRequest(query=query, docs=docs_content)
            ranked = ranker.rerank(req, top_k=k)
            return ranked, "used RerankRequest(docs=...) with 'content' only"
        except Exception as e:
            errs.append(f"RerankRequest(docs, content-only): {e}")

    try:
        ranked = ranker.rerank(query=query, documents=docs_content, top_k=k)
        return ranked, "used rerank(query, documents) with 'content' only"
    except Exception as e:
        errs.append(f"rerank(query, documents, content-only): {e}")

    try:
        ranked = ranker.rerank(query=query, docs=docs_content, top_k=k)
        return ranked, "used rerank(query, docs) with 'content' only"
    except Exception as e:
        errs.append(f"rerank(query, docs, content-only): {e}")

    try:
        ranked = ranker.rerank(query, docs_content, top_k=k)
        return ranked, "used rerank(positional, content-only)"
    except Exception as e:
        errs.append(f"rerank(positional, content-only): {e}")

    return None, " | ".join(errs)


def flashrank_rerank(
    query: str,
    articles: List[Any],
    keep_indices: List[int],
    scores: List[float],
    indices: List[int],
    result_metadata: List[Any],
    model: str = "ms-marco-TinyBERT-L-2-v2",
) -> Tuple[List[float], List[int], List[Any]]:
    """
    Rerank using FlashRank and return reordered (scores, indices, metadata).
    Falls back to original ordering if FlashRank is unavailable or incompatible.
    """
    if flashrank is None or Ranker is None:
        # FlashRank not available
        return scores, indices, result_metadata

    # Build docs for the current set of candidates
    docs = _prepare_docs(articles, keep_indices, list(indices))
    if not docs:
        return scores, indices, result_metadata

    try:
        ranker = Ranker(model_name=model)
    except Exception:
        # If model init fails, keep original order
        return scores, indices, result_metadata

    ranked, debug_msg = _call_ranker_any(ranker, query, docs, k=len(docs))
    if ranked is None:
        # Could not call FlashRank with any known signature; keep original
        return scores, indices, result_metadata

    # Build new ordering
    order_local: List[int] = []
    for item in ranked:
        _id = _extract_item_id(item)
        if _id is not None:
            order_local.append(_id)

    if not order_local:
        return scores, indices, result_metadata

    # Reorder existing tuples by the new local index order
    idx_to_tuple = {int(i): (float(s), int(i), m) for s, i, m in zip(scores, indices, result_metadata)}
    reordered = [idx_to_tuple[i] for i in order_local if i in idx_to_tuple]

    if not reordered:
        return scores, indices, result_metadata

    new_scores, new_indices, new_meta = zip(*reordered)
    return list(new_scores), list(new_indices), list(new_meta)
