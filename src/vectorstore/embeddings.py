"""
Embedding models for vector store.

This module provides embedding model wrappers for OpenAI embeddings.
"""

from abc import ABC, abstractmethod
from typing import List

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector (list of floats)
        """
        pass


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embeddings (requires API key, costs money)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embeddings.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
        """
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key)

        logger.info(f"Initialized OpenAI embeddings: {model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} documents with OpenAI")

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )

            embeddings = [item.embedding for item in response.data]

            return embeddings

        except Exception as e:
            logger.error(f"Failed to embed documents with OpenAI: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        logger.debug(f"Embedding query with OpenAI: {text[:50]}...")

        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )

            embedding = response.data[0].embedding

            return embedding

        except Exception as e:
            logger.error(f"Failed to embed query with OpenAI: {str(e)}")
            raise
