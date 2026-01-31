"""
Unit tests for text file loading.
"""

import pytest
from src.ingestion.text_loader import load_text_file


@pytest.mark.unit
class TestTextLoader:
    """Tests for text file loading."""

    def test_load_text_file(self, sample_text_file):
        """Test loading a text file."""
        doc = load_text_file(str(sample_text_file), "https://example.com")

        assert doc is not None
        assert len(doc.page_content) > 0
        assert doc.metadata["source"] == "https://example.com"
        assert doc.metadata["source_type"] == "article"
        assert doc.metadata["title"] == "Sample Article"

    def test_load_text_file_no_title(self, temp_dir):
        """Test loading text file without title metadata."""
        file_path = temp_dir / "no_title.txt"
        file_path.write_text("Just some content without metadata.")

        doc = load_text_file(str(file_path))

        assert doc is not None
        assert len(doc.page_content) > 0
        assert doc.metadata["title"] == ""

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(Exception):
            load_text_file("/nonexistent/path/file.txt")

    def test_load_empty_file(self, temp_dir):
        """Test loading empty file raises error."""
        file_path = temp_dir / "empty.txt"
        file_path.write_text("")

        with pytest.raises(ValueError):
            load_text_file(str(file_path))

    def test_load_preserves_content(self, temp_dir):
        """Test that loading preserves file content."""
        content = "Line 1\nLine 2\nLine 3"
        file_path = temp_dir / "test.txt"
        file_path.write_text(content)

        doc = load_text_file(str(file_path))

        assert content in doc.page_content
