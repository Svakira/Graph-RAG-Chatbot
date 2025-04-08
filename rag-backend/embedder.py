import os
import requests
import time
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from config import get_logger

logger = get_logger(__name__)
load_dotenv()

INFERMATIC_EMBEDDINGS_ENDPOINT = os.getenv("INFERMATIC_EMBEDDINGS_ENDPOINT")
INFERMATIC_API_KEY = os.getenv("INFERMATIC_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat-multilingual-e5-base")
EMBEDDING_RETRIES = 3
EMBEDDING_TIMEOUT = 30  # seconds


class InfermaticEmbeddings(Embeddings):
    """Wrapper for Infermatic Embeddings API service."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
        retries: int = EMBEDDING_RETRIES,
        timeout: int = EMBEDDING_TIMEOUT
    ):
        self.endpoint = endpoint or INFERMATIC_EMBEDDINGS_ENDPOINT
        self.api_key = api_key or INFERMATIC_API_KEY
        self.model = model
        self.retries = retries
        self.timeout = timeout

        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Missing Infermatic embeddings endpoint or API key. "
                "Set INFERMATIC_EMBEDDINGS_ENDPOINT and INFERMATIC_API_KEY environment variables."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.warning("Attempted to get embeddings for empty text list.")
            return []

        payload = {
            "input": texts,
            "model": self.model
        }

        for attempt in range(self.retries):
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )

                response.raise_for_status()
                response_data = response.json()

                sorted_data = sorted(response_data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]

            except requests.HTTPError as e:
                logger.error(f"Infermatic embeddings request failed (attempt {attempt+1}/{self.retries}): {e}")
                if attempt + 1 == self.retries:
                    raise ValueError(f"Failed to get embeddings after {self.retries} attempts: {e}")
                time.sleep(1 * (attempt + 1))

            except Exception as e:
                logger.error(f"Unexpected error during embedding (attempt {attempt+1}/{self.retries}): {e}")
                if attempt + 1 == self.retries:
                    raise ValueError(f"Unexpected error while getting embeddings: {e}")
                time.sleep(1 * (attempt + 1))

        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        result = self._embed([text])
        if not result:
            raise ValueError("Failed to embed query text.")
        return result[0]
