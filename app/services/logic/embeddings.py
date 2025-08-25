from __future__ import annotations

from typing import Iterable, List, Optional
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import os

try:
    import google.generativeai as genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


def _make_local_model_dir(base_dir: str, model_name: str) -> Path:
    safe_name = model_name.replace("/", "__").replace(":", "_")
    target = Path(base_dir) / safe_name
    target.mkdir(parents=True, exist_ok=True)
    return target


class GeminiEmbedder:
    """Adapter to provide a SentenceTransformer-like interface for Gemini embeddings.

    Exposes .encode(list[str], convert_to_numpy=True, normalize_embeddings=bool)
    """

    def __init__(self, model_name: str = "gemini-embeddings-001") -> None:
        if not _HAS_GENAI:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai langchain-google-genai"
            )
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_APIKEY") or os.environ.get("GOOGLE_API_KEY_GEMINI")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set for Gemini embeddings")
        genai.configure(api_key=api_key)
        self._api_model = "models/embedding-001"

    def _batch(self, texts: List[str], batch_size: int = 64) -> List[List[str]]:
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 64,
        **_: dict,
    ):
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if len(texts) == 0:
            return np.zeros((0, 768), dtype=np.float32)

        results: List[np.ndarray] = []
        for chunk in self._batch(texts, batch_size=batch_size):
            embeddings: List[np.ndarray] = []
            for t in chunk:
                resp = genai.embed_content(model=self._api_model, content=t)
                vec = np.asarray(resp["embedding"], dtype=np.float32)
                embeddings.append(vec)
            batch_arr = np.vstack(embeddings)
            if normalize_embeddings:
                batch_arr = _l2_normalize(batch_arr)
            results.append(batch_arr)

        arr = np.vstack(results)
        if convert_to_numpy:
            return arr
        return arr.tolist()


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    local_base_dir: str = "models",
):
    """Get an embedding model.

    - For Sentence-Transformers, download and cache locally when possible
    - For Gemini, return an adapter that exposes a compatible .encode()
    """
    if model_name == "gemini-embeddings-001":
        return GeminiEmbedder(model_name)

    try:
        from sentence_transformers import SentenceTransformer

        local_dir = _make_local_model_dir(local_base_dir, model_name)
        try:
            if any(local_dir.iterdir()):
                return SentenceTransformer(str(local_dir))
        except Exception:
            pass

        model = SentenceTransformer(model_name)
        try:
            model.save(str(local_dir))
        except Exception:
            pass
        return model
    except ImportError:
        raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")


def encode_texts(texts: List[str], model) -> np.ndarray:
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    emb = _to_numpy(emb)
    return _l2_normalize(emb)


class TextEmbedder:
    def __init__(
        self,
        model_name: str,
        use_sentence_transformers: bool = False,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.use_sentence_transformers = use_sentence_transformers
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sbert_model = None
        self._tokenizer = None
        self._model = None

        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._sbert_model = SentenceTransformer(self.model_name, device=self.device)
            except Exception:
                self.use_sentence_transformers = False

        if not self.use_sentence_transformers:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)

    @torch.inference_mode()
    def encode(self, texts: Iterable[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        texts_list: List[str] = [t if isinstance(t, str) else str(t) for t in texts]
        if len(texts_list) == 0:
            return np.zeros((0, 768), dtype=np.float32)

        if self.use_sentence_transformers and self._sbert_model is not None:
            embeddings = self._sbert_model.encode(
                texts_list,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32, copy=False)

        assert self._tokenizer is not None and self._model is not None

        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts_list), batch_size):
            batch = texts_list[start : start + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            token_embeddings = outputs.last_hidden_state
            sentence_embeddings = _mean_pool(token_embeddings, inputs["attention_mask"])  # [B, H]

            if normalize:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            all_embeddings.append(sentence_embeddings.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(all_embeddings)


