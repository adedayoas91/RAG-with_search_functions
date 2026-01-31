"""
Vector store module for Agentic RAG System.

This package provides embedding models and ChromaDB vector store functionality.
"""

from .embeddings import (
    EmbeddingModel,
    OpenAIEmbedding
)
from .chroma_store import ChromaVectorStore
from .parallel_embedding import (
    parallel_chunk_documents,
    parallel_embed_documents,
    shutdown_ray
)

__all__ = [
    "EmbeddingModel",
    "OpenAIEmbedding",
    "ChromaVectorStore",
    "parallel_chunk_documents",
    "parallel_embed_documents",
    "shutdown_ray",
]
