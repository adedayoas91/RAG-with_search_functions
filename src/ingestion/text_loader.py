"""
Text file loading for parsed articles.

This module provides functions to load text files that were saved
from parsed HTML articles.
"""

from pathlib import Path
from langchain_core.documents import Document

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_text_file(file_path: str, source_url: str = None) -> Document:
    """
    Load text file as a LangChain Document.

    Args:
        file_path: Path to text file
        source_url: Original URL of the article

    Returns:
        LangChain Document with text content and metadata

    Raises:
        Exception: If file cannot be read

    Example:
        >>> doc = load_text_file("/path/to/article.txt", "https://example.com")
        >>> print(f"Loaded {len(doc.page_content)} characters")
    """
    logger.info(f"Loading text file: {file_path}")

    try:
        path = Path(file_path)

        # Read file
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content:
            raise ValueError(f"Empty text file: {file_path}")

        # Extract title from first lines if available
        lines = content.split('\n')
        title = ""
        for line in lines:
            if line.startswith("Title: "):
                title = line.replace("Title: ", "").strip()
                break

        # Create Document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "source": source_url or file_path,
                "source_type": "article",
                "content_length": len(content),
                "file_path": str(path),
                "title": title
            }
        )

        logger.info(f"Successfully loaded text file: {len(content)} characters from {file_path}")

        return doc

    except Exception as e:
        logger.error(f"Failed to load text file {file_path}: {str(e)}")
        raise
