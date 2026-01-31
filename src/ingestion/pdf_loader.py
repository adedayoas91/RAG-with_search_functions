"""
PDF loading and text extraction with enhanced parsing.

This module provides functions to download and extract text from PDF files
using PyMuPDF for better structure preservation, with pypdf as fallback.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import fitz  # PyMuPDF
from pypdf import PdfReader
from langchain_core.documents import Document

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def download_pdf(url: str, output_path: Optional[str] = None, timeout: int = 30) -> str:
    """
    Download PDF from URL.

    Args:
        url: URL of PDF file
        output_path: Optional path to save PDF (if None, uses temp file)
        timeout: Request timeout in seconds

    Returns:
        Path to downloaded PDF file

    Raises:
        requests.RequestException: If download fails
    """
    logger.info(f"Downloading PDF from: {url}")

    try:
        # Set user agent to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        # Download with streaming
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        # Determine output path
        if output_path is None:
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            output_path = temp_file.name
            temp_file.close()

        # Save to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(output_path)
        logger.info(f"Downloaded PDF: {file_size / 1024:.1f} KB -> {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {str(e)}")
        raise


def extract_text_from_pdf_pymupdf(pdf_path: str) -> tuple[str, Dict[str, Any]]:
    """
    Extract text from PDF using PyMuPDF (better structure preservation).

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (extracted_text, metadata_dict)

    Raises:
        Exception: If extraction fails
    """
    logger.info(f"Extracting text from PDF with PyMuPDF: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)

        text_parts = []
        metadata = {
            "num_pages": len(doc),
            "author": doc.metadata.get("author", ""),
            "title": doc.metadata.get("title", ""),
            "subject": doc.metadata.get("subject", ""),
        }

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]

                # Extract text with better formatting
                page_text = page.get_text("text")

                if page_text.strip():
                    text_parts.append(f"\n--- Page {page_num + 1} ---\n{page_text}")

            except Exception as e:
                logger.warning(f"Failed to extract page {page_num + 1}: {str(e)}")
                continue

        doc.close()

        text = "\n".join(text_parts)
        text = clean_pdf_text(text)

        logger.info(
            f"Extracted {len(text)} characters from {metadata['num_pages']} pages"
        )

        if len(text) < 100:
            logger.warning(f"Very little text extracted ({len(text)} chars)")

        return text, metadata

    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {str(e)}")
        raise


def extract_text_from_pdf_fallback(pdf_path: str) -> tuple[str, Dict[str, Any]]:
    """
    Fallback PDF extraction using pypdf.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    logger.info(f"Using fallback pypdf extractor for: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)

        text = ""
        for page_num, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num} ---\n"
                    text += page_text
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num}: {str(e)}")
                continue

        text = clean_pdf_text(text)

        metadata = {
            "num_pages": len(reader.pages),
            "author": "",
            "title": "",
            "subject": "",
        }

        logger.info(f"Extracted {len(text)} characters from {len(reader.pages)} pages")

        return text, metadata

    except Exception as e:
        logger.error(f"Fallback extraction failed: {str(e)}")
        raise


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file (tries PyMuPDF first, falls back to pypdf).

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text content

    Raises:
        Exception: If all extraction methods fail
    """
    try:
        text, _ = extract_text_from_pdf_pymupdf(pdf_path)
        return text
    except Exception as e:
        logger.warning(f"PyMuPDF failed, trying fallback: {str(e)}")
        try:
            text, _ = extract_text_from_pdf_fallback(pdf_path)
            return text
        except Exception as e2:
            logger.error(f"All extraction methods failed: {str(e2)}")
            raise


def clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    import re

    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Replace multiple newlines with double newline
    text = re.sub(r'\n\n+', '\n\n', text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def load_pdf_from_file(file_path: str, source_url: Optional[str] = None) -> Document:
    """
    Load PDF from local file as a LangChain Document.

    Args:
        file_path: Path to local PDF file
        source_url: Optional original URL of the PDF

    Returns:
        LangChain Document with extracted text and metadata

    Raises:
        Exception: If extraction fails

    Example:
        >>> doc = load_pdf_from_file("/path/to/paper.pdf")
        >>> print(f"Loaded {len(doc.page_content)} characters")
    """
    logger.info(f"Loading PDF from file: {file_path}")

    try:
        # Extract text with metadata
        text, pdf_metadata = extract_text_from_pdf_pymupdf(file_path)

        if not text:
            # Try fallback
            text, pdf_metadata = extract_text_from_pdf_fallback(file_path)

        if not text:
            raise ValueError(f"No text extracted from PDF: {file_path}")

        # Create Document with metadata
        doc = Document(
            page_content=text,
            metadata={
                "source": source_url or file_path,
                "source_type": "pdf",
                "content_length": len(text),
                "file_path": file_path,
                "num_pages": pdf_metadata.get("num_pages", 0),
                "title": pdf_metadata.get("title", ""),
                "author": pdf_metadata.get("author", ""),
            }
        )

        logger.info(f"Successfully loaded PDF: {len(text)} characters from {file_path}")

        return doc

    except Exception as e:
        logger.error(f"Failed to load PDF file {file_path}: {str(e)}")
        raise


def load_pdf_source(url: str, cleanup: bool = True) -> Document:
    """
    Load PDF from URL as a LangChain Document.

    This is the main function to use for loading PDFs into the RAG system.

    Args:
        url: URL of PDF file
        cleanup: Whether to delete temporary file after extraction

    Returns:
        LangChain Document with extracted text and metadata

    Raises:
        Exception: If download or extraction fails

    Example:
        >>> doc = load_pdf_source("https://example.com/paper.pdf")
        >>> print(f"Loaded {len(doc.page_content)} characters")
        >>> print(doc.metadata)
    """
    logger.info(f"Loading PDF source: {url}")

    pdf_path = None
    try:
        # Download PDF
        pdf_path = download_pdf(url)

        # Load from downloaded file
        doc = load_pdf_from_file(pdf_path, source_url=url)

        return doc

    except Exception as e:
        logger.error(f"Failed to load PDF source {url}: {str(e)}")
        raise

    finally:
        # Cleanup temp file if requested
        if cleanup and pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
                logger.debug(f"Cleaned up temporary file: {pdf_path}")
            except:
                pass
