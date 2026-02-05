"""
Docling-based document processor for enhanced content extraction.

This module uses Docling to parse and extract content from various document formats
before the chunking process begins. Docling provides advanced document understanding
capabilities for better text extraction.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

from src.utils import get_logger

logger = get_logger(__name__)

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.backend.pypdf_backend import PyPdfDocumentBackend
    DOCLING_AVAILABLE = True
except ImportError:
    logger.warning("Docling not available. Using fallback document processing.")
    DOCLING_AVAILABLE = False
    DocumentConverter = None


class DoclingDocumentProcessor:
    """
    Enhanced document processor using Docling for advanced content extraction.
    """

    def __init__(self):
        """Initialize the Docling document processor."""
        self.converter = None
        if DOCLING_AVAILABLE:
            # Configure pipeline options for better PDF processing
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for scanned documents
            pipeline_options.do_table_structure = True  # Extract table structures

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pipeline_options,
                    InputFormat.DOCX: pipeline_options,
                }
            )
            logger.info("Docling document processor initialized")
        else:
            logger.warning("Docling not available, falling back to basic processing")

    def process_document(self, file_path: Path) -> Optional[Document]:
        """
        Process a single document using Docling for enhanced extraction.

        Args:
            file_path: Path to the document file

        Returns:
            Processed Document object or None if processing failed
        """
        if not DOCLING_AVAILABLE or not self.converter:
            logger.warning("Docling not available, skipping enhanced processing")
            return None

        try:
            logger.info(f"Processing document with Docling: {file_path}")

            # Convert document using Docling
            result = self.converter.convert(str(file_path))

            if not result or not result.document:
                logger.warning(f"Docling conversion failed for {file_path}")
                return None

            # Extract text content
            text_content = result.document.export_to_markdown()

            # Extract metadata
            metadata = {
                "source": str(file_path),
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix.lower(),
                "processed_by": "docling",
                "docling_format": str(result.document.format) if hasattr(result.document, 'format') else 'unknown'
            }

            # Add additional metadata if available
            if hasattr(result.document, 'properties'):
                doc_props = result.document.properties
                if hasattr(doc_props, 'title') and doc_props.title:
                    metadata["title"] = doc_props.title
                if hasattr(doc_props, 'author') and doc_props.author:
                    metadata["author"] = doc_props.author
                if hasattr(doc_props, 'subject') and doc_props.subject:
                    metadata["subject"] = doc_props.subject

            # Create LangChain Document
            document = Document(
                page_content=text_content,
                metadata=metadata
            )

            logger.info(f"Successfully processed {file_path} with Docling")
            return document

        except Exception as e:
            logger.error(f"Error processing {file_path} with Docling: {str(e)}")
            return None

    def process_documents_batch(self, file_paths: List[Path]) -> List[Document]:
        """
        Process multiple documents using Docling.

        Args:
            file_paths: List of paths to document files

        Returns:
            List of processed Document objects
        """
        processed_documents = []

        for file_path in file_paths:
            logger.info(f"Processing {file_path.name}...")

            # Try Docling processing first
            doc = self.process_document(file_path)

            if doc:
                processed_documents.append(doc)
                logger.info(f"✅ Processed {file_path.name}")
            else:
                logger.warning(f"❌ Failed to process {file_path.name} with Docling")

        logger.info(f"Docling processing complete: {len(processed_documents)}/{len(file_paths)} documents processed")
        return processed_documents


def create_docling_processor() -> DoclingDocumentProcessor:
    """
    Factory function to create a Docling document processor.

    Returns:
        Configured DoclingDocumentProcessor instance
    """
    return DoclingDocumentProcessor()


def is_docling_available() -> bool:
    """
    Check if Docling is available for document processing.

    Returns:
        True if Docling can be used, False otherwise
    """
    return DOCLING_AVAILABLE