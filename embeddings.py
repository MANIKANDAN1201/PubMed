from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Try to import optional dependencies
try:
    import google.generativeai as genai  # type: ignore
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""
    name: str
    model_id: str
    dimensions: int
    is_api_based: bool = False
    requires_auth: bool = False
    auth_env_var: Optional[str] = None
    use_sentence_transformers: bool = False
    max_seq_length: int = 512

# Supported models configuration
SUPPORTED_MODELS = {
    "pubmedbert": EmbeddingModelConfig(
        name="PubMedBERT",
        model_id="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        dimensions=768,
        use_sentence_transformers=False
    ),
    "biobert": EmbeddingModelConfig(
        name="BioBERT",
        model_id="dmis-lab/biobert-base-cased-v1.1",
        dimensions=768,
        use_sentence_transformers=False
    ),
    "minilm": EmbeddingModelConfig(
        name="MiniLM",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        use_sentence_transformers=True
    ),
    "gemini": EmbeddingModelConfig(
        name="Gemini",
        model_id="models/embedding-001",
        dimensions=768,
        is_api_based=True,
        requires_auth=True,
        auth_env_var="GOOGLE_API_KEY"
    ),
    "jina": EmbeddingModelConfig(
        name="Jina",
        model_id="jinaai/jina-embeddings-v2-base-en",
        dimensions=768,
        use_sentence_transformers=True
    )
}


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


class PubMedBERTEmbedder:
    """Adapter to provide a SentenceTransformer-like interface for PubMedBERT embeddings.
    Exposes .encode(list[str], convert_to_numpy=True, normalize_embeddings=bool)
    """

    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract") -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device).eval()
        self.dimensions = 768  # Standard for BERT-base models

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        **_: dict,
    ) -> np.ndarray:
        """Encode texts using PubMedBERT with mean pooling."""
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling on last hidden states
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())

        embeddings = np.vstack(all_embeddings)
        return embeddings.astype(np.float32)


class BioBERTEmbedder(PubMedBERTEmbedder):
    """Adapter to provide a SentenceTransformer-like interface for BioBERT embeddings."""
    
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1") -> None:
        super().__init__(model_name)
        self.dimensions = 768  # BioBERT uses BERT-base architecture


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
        # Map friendly name to actual API model id
        # Keeping older naming consistent with user's request
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
            # google.generativeai embed_content supports a single string or list via loop
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

        # Try loading from local cache first
        local_dir = _make_local_model_dir(local_base_dir, model_name)
        try:
            if any(local_dir.iterdir()):
                return SentenceTransformer(str(local_dir))
        except Exception:
            pass

        # Otherwise load from hub and persist locally for future runs
        model = SentenceTransformer(model_name)
        try:
            model.save(str(local_dir))
        except Exception:
            pass
        return model
    except ImportError:
        raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")


def get_available_models() -> Dict[str, Dict[str, Union[str, int]]]:
    """Get information about all available models."""
    return {
        name: {
            'name': config.name,
            'dimensions': config.dimensions,
            'is_api_based': config.is_api_based,
            'requires_auth': config.requires_auth
        }
        for name, config in SUPPORTED_MODELS.items()
    }

def encode_texts(texts: List[str], model) -> np.ndarray:
    """Encode texts using a model - simplified interface matching reference code"""
    if hasattr(model, 'encode'):
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    elif hasattr(model, 'embed_texts'):
        emb = model.embed_texts(texts)
    else:
        raise ValueError("Model must have either 'encode' or 'embed_texts' method")
    
    emb = _to_numpy(emb)
    return _l2_normalize(emb)


class TextEmbedder:
    def __init__(
        self,
        model_name: str,
        use_sentence_transformers: Optional[bool] = None,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        """Initialize the text embedder with the specified model.
        
        Args:
            model_name: Name of the model to use (must be a key in SUPPORTED_MODELS)
            use_sentence_transformers: Whether to use sentence-transformers (auto-detected if None)
            device: Device to run the model on (auto-detected if None)
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = SUPPORTED_MODELS.get(model_name)
        
        if self.config is None:
            # Fallback for custom models not in SUPPORTED_MODELS
            self.config = EmbeddingModelConfig(
                name=model_name.split('/')[-1],
                model_id=model_name,
                dimensions=768,  # Default, will be updated after model loading
                use_sentence_transformers=use_sentence_transformers or False
            )
        
        # Initialize model based on configuration
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on configuration."""
        if self.config.is_api_based:
            if self.config.requires_auth and self.config.auth_env_var:
                api_key = os.getenv(self.config.auth_env_var)
                if not api_key:
                    raise ValueError(f"API key not found in {self.config.auth_env_var}")
                if self.config.model_id == "models/embedding-001" and _HAS_GENAI:
                    genai.configure(api_key=api_key)
            return  # API models are initialized on demand
        
        if self.config.use_sentence_transformers and _HAS_SENTENCE_TRANSFORMERS:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.model_id, device=self.device)
            self.config.dimensions = self.model.get_sentence_embedding_dimension()
        else:
            # Special handling for PubMedBERT and similar models
            if 'pubmedbert' in self.model_name.lower() or 'biomednlp' in self.config.model_id.lower():
                # Ensure we're using the correct tokenizer for PubMedBERT
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
                # Load model with output_hidden_states=True to access all layers
                self.model = AutoModel.from_pretrained(
                    self.config.model_id,
                    output_hidden_states=True
                )
            else:
                # Standard transformer model loading
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
                self.model = AutoModel.from_pretrained(self.config.model_id)
            
            self.model.eval()
            self.model.to(self.device)
            # Get embedding dimension from config or model
            if hasattr(self.model.config, 'hidden_size'):
                self.config.dimensions = self.model.config.hidden_size
    
    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.config.dimensions

    def _embed_with_api(self, texts: List[str]) -> np.ndarray:
        """Embed texts using an external API."""
        if self.config.model_id == "models/embedding-001" and _HAS_GENAI:
            chunk_size = 50  # Process in chunks to avoid rate limits
            embeddings = []
            
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i+chunk_size]
                response = genai.embed_content(
                    model=self.config.model_id,
                    content=chunk,
                    task_type="retrieval_document"
                )
                embeddings.extend(response['embedding'])
            
            return np.array(embeddings)
        else:
            raise NotImplementedError(f"API-based embedding not implemented for {self.config.model_id}")

    def _embed_with_transformers(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Embed texts using a HuggingFace transformer model."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                # Use mean pooling for sentence embeddings
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

    @torch.inference_mode()
    def encode(self, texts: Iterable[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize the embeddings
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        texts_list = [t if isinstance(t, str) else str(t) for t in texts]
        if not texts_list:
            return np.zeros((0, self.get_embedding_dimensions()), dtype=np.float32)
        
        # Handle API-based models
        if self.config.is_api_based:
            return self._embed_with_api(texts_list)
        
        # Handle local models
        if self.config.use_sentence_transformers and hasattr(self, 'model'):
            embeddings = self.model.encode(
                texts_list,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32, copy=False)
        
        # Handle standard transformers models
        if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
            embeddings = self._embed_with_transformers(texts_list, batch_size)
            if normalize:
                embeddings = _l2_normalize(embeddings)
            return embeddings
        
        raise RuntimeError("No valid model or tokenizer found for encoding")
