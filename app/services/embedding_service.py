from typing import List
import numpy as np
from .logic.embeddings import TextEmbedder, GeminiEmbedder


_models: dict[str, TextEmbedder] = {}


def get_embedder(model_name: str, use_sentence_transformers: bool = False):
    # Special-case Gemini
    if model_name in {"gemini", "gemini-embeddings-001", "models/embedding-001"}:
        key = "gemini"
        if key not in _models:
            _models[key] = GeminiEmbedder("gemini-embeddings-001")
        return _models[key]
    key = f"{model_name}|{use_sentence_transformers}"
    if key not in _models:
        _models[key] = TextEmbedder(model_name=model_name, use_sentence_transformers=use_sentence_transformers)
    return _models[key]


def embed_texts(texts: List[str], model_name: str, use_sentence_transformers: bool = False) -> List[List[float]]:
    model = get_embedder(model_name, use_sentence_transformers)
    arr = model.encode(texts, batch_size=16, normalize=True)
    return np.asarray(arr, dtype=np.float32).tolist()


