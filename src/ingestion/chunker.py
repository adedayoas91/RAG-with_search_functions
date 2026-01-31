"""
Document chunking for RAG ingestion.

This module provides functions to split documents into smaller chunks
suitable for embedding and retrieval.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter to intelligently split on
    paragraph and sentence boundaries when possible.

    Args:
        documents: List of LangChain Document objects
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of chunked Document objects with preserved metadata

    Example:
        >>> docs = [Document(page_content="Long text...", metadata={"source": "url"})]
        >>> chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        >>> print(f"Split into {len(chunks)} chunks")
    """
    if not documents:
        logger.warning("No documents to chunk")
        return []

    logger.info(
        f"Chunking {len(documents)} documents "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on natural boundaries
        )

        chunks = splitter.split_documents(documents)

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Log stats
        if chunks:
            chunk_lengths = [len(c.page_content) for c in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            logger.debug(
                f"Chunk stats - Min: {min(chunk_lengths)}, "
                f"Max: {max(chunk_lengths)}, Avg: {avg_length:.0f}"
            )

        return chunks

    except Exception as e:
        logger.error(f"Failed to chunk documents: {str(e)}")
        raise