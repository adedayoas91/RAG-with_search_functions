"""
Unit tests for PDF loading and extraction.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.ingestion.pdf_loader import (
    extract_text_from_pdf_pymupdf,
    load_pdf_from_file,
    load_pdf_source
)


@pytest.mark.unit
class TestPDFLoader:
    """Tests for PDF loading functionality."""

    @patch('fitz.open')
    def test_extract_text_from_pdf_pymupdf(self, mock_fitz_open):
        """Test PyMuPDF text extraction."""
        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Test PDF", "author": "Test Author"}
        mock_doc.__len__.return_value = 2

        # Mock pages
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "This is page 1 content."
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "This is page 2 content."

        mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
        mock_fitz_open.return_value = mock_doc

        text, metadata = extract_text_from_pdf_pymupdf("test.pdf")

        assert "page 1 content" in text
        assert "page 2 content" in text
        assert metadata["num_pages"] == 2
        assert metadata["title"] == "Test PDF"
        assert metadata["author"] == "Test Author"

    @patch('fitz.open')
    def test_extract_text_from_empty_pdf(self, mock_fitz_open):
        """Test handling of empty PDF."""
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_doc.__len__.return_value = 0
        mock_fitz_open.return_value = mock_doc

        text, metadata = extract_text_from_pdf_pymupdf("empty.pdf")

        assert text == ""
        assert metadata["num_pages"] == 0

    @patch('fitz.open')
    def test_load_pdf_from_file(self, mock_fitz_open):
        """Test loading PDF from local file."""
        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Test PDF"}
        mock_doc.__len__.return_value = 1

        mock_page = MagicMock()
        mock_page.get_text.return_value = "PDF content here"
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        doc = load_pdf_from_file("/path/to/test.pdf", source_url="https://example.com/test.pdf")

        assert doc is not None
        assert "PDF content" in doc.page_content
        assert doc.metadata["source"] == "https://example.com/test.pdf"
        assert doc.metadata["source_type"] == "pdf"
        assert doc.metadata["num_pages"] == 1

    @patch('fitz.open')
    def test_load_pdf_from_file_without_url(self, mock_fitz_open):
        """Test loading PDF without source URL."""
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_doc.__len__.return_value = 1

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Content"
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        doc = load_pdf_from_file("/path/to/test.pdf")

        assert doc.metadata["source"] == "/path/to/test.pdf"

    @patch('fitz.open')
    def test_pdf_extraction_preserves_structure(self, mock_fitz_open):
        """Test that PDF extraction preserves page structure."""
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_doc.__len__.return_value = 2

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "First page"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Second page"

        mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
        mock_fitz_open.return_value = mock_doc

        text, _ = extract_text_from_pdf_pymupdf("test.pdf")

        assert "--- Page 1 ---" in text
        assert "--- Page 2 ---" in text
        assert text.index("First page") < text.index("Second page")

    @patch('requests.get')
    @patch('fitz.open')
    def test_load_pdf_source_with_download(self, mock_fitz_open, mock_get):
        """Test loading PDF from URL with download."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"%PDF-1.4 test content"
        mock_get.return_value = mock_response

        # Mock PDF extraction
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_doc.__len__.return_value = 1
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Downloaded PDF content"
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        doc = load_pdf_source("https://example.com/paper.pdf")

        assert doc is not None
        assert "Downloaded PDF content" in doc.page_content
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_load_pdf_source_download_failure(self, mock_get):
        """Test handling of failed PDF download."""
        mock_get.side_effect = Exception("Connection error")

        with pytest.raises(Exception):
            load_pdf_source("https://example.com/paper.pdf")

    @patch('fitz.open')
    def test_load_pdf_with_special_characters(self, mock_fitz_open):
        """Test loading PDF with special characters in text."""
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_doc.__len__.return_value = 1

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Content with €, ñ, and 中文"
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        doc = load_pdf_from_file("test.pdf")

        assert "€" in doc.page_content
        assert "ñ" in doc.page_content
        assert "中文" in doc.page_content
