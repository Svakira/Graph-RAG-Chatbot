import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from config import get_logger

logger = get_logger(__name__)
load_dotenv()

AZURE_AI_EMBEDDINGS_ENDPOINT = os.getenv("AZURE_AI_EMBEDDINGS_ENDPOINT")
AZURE_AI_EMBEDDINGS_KEY = os.getenv("AZURE_AI_EMBEDDINGS_KEY")
EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_RETRIES = 3
EMBEDDING_TIMEOUT = 30  # seconds



class AzureEmbeddings(Embeddings):
    """Wrapper for Azure AI Embeddings service."""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
        retries: int = EMBEDDING_RETRIES,
        timeout: int = EMBEDDING_TIMEOUT
    ):
        """Initialize with Azure AI Embeddings service parameters."""
        self.endpoint = endpoint or AZURE_AI_EMBEDDINGS_ENDPOINT
        self.api_key = api_key or AZURE_AI_EMBEDDINGS_KEY
        self.model = model
        self.retries = retries
        self.timeout = timeout
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Missing Azure AI Embeddings endpoint or API key. "
                "Set AZURE_AI_EMBEDDINGS_ENDPOINT and AZURE_AI_EMBEDDINGS_KEY environment variables."
            )
        
        self.client = EmbeddingsClient(
            endpoint=self.endpoint, 
            credential=AzureKeyCredential(self.api_key)
        )
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        if not texts:
            logger.warning("Attempted to get embeddings for empty text list.")
            return []
        
        for attempt in range(self.retries):
            try:
                response = self.client.embed(
                    input=texts, 
                    model=self.model
                )
                
                # Sort by index 
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]
                
            except HttpResponseError as e:
                logger.error(f"Azure AI Embeddings request failed (attempt {attempt+1}/{self.retries}): {e}")
                if attempt + 1 == self.retries:
                    raise ValueError(f"Failed to get embeddings after {self.retries} attempts: {e}")
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error during embedding (attempt {attempt+1}/{self.retries}): {e}")
                if attempt + 1 == self.retries:
                    raise ValueError(f"Unexpected error while getting embeddings: {e}")
                time.sleep(1 * (attempt + 1))
        
        return []  
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents."""
        return self._embed(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query text."""
        result = self._embed([text])
        if not result:
            raise ValueError("Failed to embed query text.")
        return result[0]

