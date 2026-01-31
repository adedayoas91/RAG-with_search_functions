"""
Parallel document processing using Ray.

This module provides parallelized chunking and embedding using Ray for improved performance.
"""

import ray
from typing import List
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.vectorstore.embeddings import EmbeddingModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def ensure_ray_initialized():
    """Initialize Ray if not already initialized."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=None)
        logger.info("Ray initialized for parallel processing")


@ray.remote
def chunk_document_batch(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Ray remote function to chunk a batch of documents.

    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked Document objects
    """
    from src.ingestion.chunker import chunk_documents
    return chunk_documents(documents, chunk_size, chunk_overlap)


@ray.remote
class ParallelEmbedder:
    """Ray actor for parallel embedding generation."""

    def __init__(self, embedding_model_config: dict):
        """
        Initialize embedder with model configuration.

        Args:
            embedding_model_config: Dict with 'type', 'api_key', and 'model' keys
        """
        from src.vectorstore.embeddings import OpenAIEmbedding

        if embedding_model_config['type'] == 'openai':
            self.model = OpenAIEmbedding(
                api_key=embedding_model_config['api_key'],
                model=embedding_model_config['model']
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {embedding_model_config['type']}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        return self.model.embed_documents(texts)


def parallel_chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    num_workers: int = 4
) -> List[Document]:
    """
    Chunk documents in parallel using Ray.

    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        num_workers: Number of parallel workers

    Returns:
        List of chunked Document objects
    """
    ensure_ray_initialized()

    logger.info(f"Chunking {len(documents)} documents in parallel with {num_workers} workers")

    # Split documents into batches
    batch_size = max(1, len(documents) // num_workers)
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    # Process batches in parallel
    futures = [
        chunk_document_batch.remote(batch, chunk_size, chunk_overlap)
        for batch in batches
    ]

    # Collect results
    results = ray.get(futures)

    # Flatten results
    all_chunks = []
    for batch_chunks in results:
        all_chunks.extend(batch_chunks)

    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

    return all_chunks


def parallel_embed_documents(
    chunks: List[Document],
    embedding_model: EmbeddingModel,
    api_key: str,
    model_name: str = "text-embedding-3-small",
    num_workers: int = 4,
    batch_size: int = 100
) -> List[List[float]]:
    """
    Generate embeddings for chunks in parallel using Ray.

    Args:
        chunks: List of Document chunks
        embedding_model: EmbeddingModel instance (not used, kept for compatibility)
        api_key: OpenAI API key
        model_name: Embedding model name
        num_workers: Number of parallel workers
        batch_size: Number of texts per batch

    Returns:
        List of embedding vectors
    """
    ensure_ray_initialized()

    logger.info(f"Generating embeddings for {len(chunks)} chunks in parallel with {num_workers} workers")

    # Extract texts
    texts = [chunk.page_content for chunk in chunks]

    # Create embedding model config
    model_config = {
        'type': 'openai',
        'api_key': api_key,
        'model': model_name
    }

    # Create Ray actors
    embedders = [ParallelEmbedder.remote(model_config) for _ in range(num_workers)]

    # Split texts into batches
    text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    # Distribute batches across workers
    futures = []
    for i, batch in enumerate(text_batches):
        embedder = embedders[i % num_workers]
        futures.append(embedder.embed_batch.remote(batch))

    # Collect results
    results = ray.get(futures)

    # Flatten results
    all_embeddings = []
    for batch_embeddings in results:
        all_embeddings.extend(batch_embeddings)

    logger.info(f"Generated {len(all_embeddings)} embeddings")

    return all_embeddings


def shutdown_ray():
    """Shutdown Ray if initialized."""
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown")
