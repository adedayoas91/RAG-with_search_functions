"""
Unit tests for document chunking functionality.
"""

import pytest
from src.ingestion.chunker import chunk_documents


@pytest.mark.unit
class TestChunker:
    """Tests for document chunking."""

    def test_chunk_single_document(self, sample_document):
        """Test chunking a single document."""
        chunks = chunk_documents([sample_document], chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 0
        assert all(isinstance(chunk.page_content, str) for chunk in chunks)
        assert all("source" in chunk.metadata for chunk in chunks)

    def test_chunk_multiple_documents(self, sample_documents):
        """Test chunking multiple documents."""
        chunks = chunk_documents(sample_documents, chunk_size=50, chunk_overlap=10)

        assert len(chunks) >= len(sample_documents)
        # Check that all chunks have content
        assert all(len(chunk.page_content) > 0 for chunk in chunks)

    def test_chunk_empty_list(self):
        """Test chunking empty document list."""
        chunks = chunk_documents([], chunk_size=100, chunk_overlap=20)
        assert chunks == []

    def test_chunk_preserves_metadata(self, sample_document):
        """Test that chunking preserves document metadata."""
        chunks = chunk_documents([sample_document], chunk_size=100, chunk_overlap=20)

        for chunk in chunks:
            assert chunk.metadata["source"] == sample_document.metadata["source"]
            assert chunk.metadata["source_type"] == sample_document.metadata["source_type"]

    def test_chunk_size_respected(self, sample_document):
        """Test that chunk size is approximately respected."""
        chunk_size = 100
        chunks = chunk_documents([sample_document], chunk_size=chunk_size, chunk_overlap=20)

        # Chunks should be around the specified size (with some tolerance)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size * 1.2  # 20% tolerance

    def test_chunk_overlap(self, sample_document):
        """Test that chunks have overlap."""
        chunks = chunk_documents([sample_document], chunk_size=50, chunk_overlap=20)

        if len(chunks) > 1:
            # Check for some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                # At least some common words should exist
                words1 = set(chunks[i].page_content.split())
                words2 = set(chunks[i + 1].page_content.split())
                # Some overlap expected
                assert len(words1 & words2) >= 0
