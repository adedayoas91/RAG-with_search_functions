"""
Unit tests for RAG answer generation.
"""

import pytest
from unittest.mock import Mock
from src.generation.answer_generator import RAGAnswerGenerator, GeneratedAnswer


@pytest.mark.unit
class TestAnswerGenerator:
    """Tests for RAG answer generator."""

    def test_initialization(self, mock_openai_client):
        """Test initializing answer generator."""
        generator = RAGAnswerGenerator(
            client=mock_openai_client,
            model="gpt-4o",
            cost_tracker=None
        )

        assert generator.client == mock_openai_client
        assert generator.model == "gpt-4o"

    def test_generate_answer(self, mock_openai_client, sample_documents):
        """Test generating answer with context."""
        generator = RAGAnswerGenerator(client=mock_openai_client)

        answer = generator.generate_answer(
            query="What is machine learning?",
            context_documents=sample_documents,
            temperature=0.2,
            max_tokens=500
        )

        assert isinstance(answer, GeneratedAnswer)
        assert len(answer.answer) > 0
        assert len(answer.sources) > 0
        assert answer.tokens_used > 0
        assert answer.model == "gpt-4o-mini"
        assert "## Sources" in answer.answer

    def test_generate_answer_with_cost_tracking(
        self, mock_openai_client, sample_documents, mock_cost_tracker
    ):
        """Test generating answer with cost tracking."""
        generator = RAGAnswerGenerator(
            client=mock_openai_client,
            cost_tracker=mock_cost_tracker
        )

        answer = generator.generate_answer(
            query="What is AI?",
            context_documents=sample_documents
        )

        assert answer.cost >= 0
        mock_cost_tracker.track_openai_call.assert_called_once()

    def test_generate_answer_empty_context(self, mock_openai_client):
        """Test generating answer with empty context."""
        generator = RAGAnswerGenerator(client=mock_openai_client)

        answer = generator.generate_answer(
            query="What is machine learning?",
            context_documents=[],
            temperature=0.2,
            max_tokens=500
        )

        assert isinstance(answer, GeneratedAnswer)
        assert len(answer.sources) == 0

    def test_format_context(self, mock_openai_client, sample_documents):
        """Test context formatting."""
        generator = RAGAnswerGenerator(client=mock_openai_client)

        context, sources = generator._format_context(sample_documents, max_context_length=5000)

        assert isinstance(context, str)
        assert len(sources) == len(sample_documents)
        assert "[1]" in context
        assert all(isinstance(s, str) for s in sources)

    def test_append_sources_list(self, mock_openai_client):
        """Test appending sources list to answer."""
        generator = RAGAnswerGenerator(client=mock_openai_client)

        answer = "This is the answer with citations [1], [2]."
        sources = ["https://example.com/1", "https://example.com/2"]

        result = generator._append_sources_list(answer, sources)

        assert "## Sources" in result
        assert "[1] https://example.com/1" in result
        assert "[2] https://example.com/2" in result

    def test_context_truncation(self, mock_openai_client):
        """Test that context is truncated when too long."""
        from langchain_core.documents import Document

        generator = RAGAnswerGenerator(client=mock_openai_client)

        # Create a very long document
        long_docs = [
            Document(
                page_content="x" * 5000,
                metadata={"source": f"https://example.com/{i}"}
            )
            for i in range(10)
        ]

        context, sources = generator._format_context(long_docs, max_context_length=8000)

        # Context should be truncated
        assert len(context) <= 8000
        # Not all sources should be included
        assert len(sources) < len(long_docs)
