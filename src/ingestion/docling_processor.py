"""
Enhanced document processor for improved content extraction.

This module provides enhanced document processing using available libraries
(Docling if available, otherwise PyMuPDF and fallback methods) for better
text extraction from various document formats before chunking.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

from src.utils import get_logger

logger = get_logger(__name__)

# Check for available enhanced processing libraries
try:
    import fitz  # PyMuPDF for enhanced PDF processing
    PYMUPDF_AVAILABLE = True
    logger.info("PyMuPDF available for enhanced PDF processing")
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available for enhanced PDF processing")

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
    logger.info("Docling available for advanced document processing")
except ImportError:
    DOCLING_AVAILABLE = False
    logger.info("Docling not available, using fallback processing")


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
        Process a single document using enhanced extraction methods.

        Args:
            file_path: Path to the document file

        Returns:
            Processed Document object or None if processing failed
        """
        try:
            logger.info(f"Processing document: {file_path}")

            # Try Docling first if available
            if DOCLING_AVAILABLE and self.converter:
                return self._process_with_docling(file_path)
            else:
                # Fall back to enhanced processing with available libraries
                return self._process_with_fallback(file_path)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    def _process_with_docling(self, file_path: Path) -> Optional[Document]:
        """Process document using Docling."""
        try:
            # Convert document using Docling
            result = self.converter.convert(str(file_path))

            if not result or not result.document:
                logger.warning(f"Docling conversion failed for {file_path}")
                return self._process_with_fallback(file_path)

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
            logger.error(f"Docling processing failed for {file_path}: {str(e)}")
            return self._process_with_fallback(file_path)

    def _process_with_fallback(self, file_path: Path) -> Optional[Document]:
        """Process document using fallback methods (existing PDF/text processing)."""
        try:
            from src.ingestion.pdf_loader import load_pdf_from_file
            from src.ingestion.text_loader import load_text_file

            logger.info(f"Using fallback processing for {file_path}")

            if file_path.suffix.lower() == '.pdf':
                if PYMUPDF_AVAILABLE:
                    # Use enhanced PDF processing if available
                    return self._enhanced_pdf_processing(file_path)
                else:
                    # Fall back to basic PDF processing
                    doc = load_pdf_from_file(
                        str(file_path),
                        source_url=f"file://{file_path.absolute()}"
                    )
                    if doc:
                        # Add enhanced metadata
                        doc.metadata.update({
                            "processed_by": "fallback_pdf",
                            "enhanced_processing": False
                        })
                    return doc

            elif file_path.suffix.lower() == '.txt':
                doc = load_text_file(
                    str(file_path),
                    source_url=f"file://{file_path.absolute()}"
                )
                if doc:
                    doc.metadata.update({
                        "processed_by": "fallback_text",
                        "enhanced_processing": False
                    })
                return doc

            else:
                logger.warning(f"Unsupported file type for enhanced processing: {file_path.suffix}")
                return None

        except Exception as e:
            logger.error(f"Fallback processing failed for {file_path}: {str(e)}")
            return None

    def _enhanced_pdf_processing(self, file_path: Path) -> Optional[Document]:
        """Enhanced PDF processing using PyMuPDF."""
        try:
            logger.info(f"Enhanced PDF processing for {file_path}")

            # Open PDF with PyMuPDF
            doc = fitz.open(str(file_path))

            # Extract text with better formatting
            text_content = ""
            metadata = {}

            # Get document info
            doc_info = doc.metadata
            if doc_info:
                metadata.update({
                    "title": doc_info.get("title", ""),
                    "author": doc_info.get("author", ""),
                    "subject": doc_info.get("subject", ""),
                    "creator": doc_info.get("creator", ""),
                    "producer": doc_info.get("producer", ""),
                })

            # Extract text from all pages
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()

                # Add page separator
                if page_num > 0:
                    text_content += f"\n\n--- Page {page_num + 1} ---\n\n"
                text_content += page_text

            doc.close()

            # Create enhanced metadata
            enhanced_metadata = {
                "source": str(file_path),
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_type": "pdf",
                "processed_by": "enhanced_pymupdf",
                "enhanced_processing": True,
                "total_pages": len(doc),
                **metadata
            }

            # Create LangChain Document
            document = Document(
                page_content=text_content,
                metadata=enhanced_metadata
            )

            logger.info(f"Enhanced PDF processing completed for {file_path}")
            return document

        except Exception as e:
            logger.error(f"Enhanced PDF processing failed for {file_path}: {str(e)}")
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


def is_enhanced_processing_available() -> bool:
    """
    Check if enhanced document processing is available.

    Returns:
        True if any enhanced processing (Docling or PyMuPDF) is available
    """
    return DOCLING_AVAILABLE or PYMUPDF_AVAILABLE


def is_docling_available() -> bool:
    """
    Check if Docling is available for document processing.

    Returns:
        True if Docling can be used, False otherwise
    """
    return DOCLING_AVAILABLE