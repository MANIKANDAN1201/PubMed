from typing import List, Optional, Dict, Any
import logging
import numpy as np

# Import existing functionality with fallbacks
try:
    from embeddings import TextEmbedder
except ImportError:
    TextEmbedder = None

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for text embedding generation and management"""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        self.model_name = model_name
        self.embedder = None
        
        if TextEmbedder:
            try:
                self.embedder = TextEmbedder(model_name=model_name)
                logger.info(f"Initialized embedding service with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize embedder: {e}")
        else:
            logger.warning("TextEmbedder not available - install transformers and torch")
    
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Encode a list of texts to embeddings
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        if not self.embedder:
            logger.error("Embedder not available")
            return None
        
        try:
            embeddings = self.embedder.encode(texts, batch_size=batch_size, normalize=normalize)
            logger.info(f"Encoded {len(texts)} texts to embeddings of shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            return None
    
    def encode_single_text(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Encode a single text to embedding
        
        Args:
            text: Text string
            normalize: Whether to normalize embedding
            
        Returns:
            Numpy array of embedding or None if failed
        """
        return self.encode_texts([text], batch_size=1, normalize=normalize)
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings"""
        if not self.embedder:
            return None
        
        try:
            # Try to get dimension from a sample encoding
            sample_embedding = self.encode_single_text("sample text")
            if sample_embedding is not None:
                return sample_embedding.shape[1]
        except Exception:
            pass
        
        return None
    
    def is_available(self) -> bool:
        """Check if embedding service is available"""
        return self.embedder is not None
