"""
Shared test fixtures and configuration for pytest.

This module provides common fixtures, mocks, and utilities used across all tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document

# Sample test data
SAMPLE_TEXT = """
This is a sample article about machine learning.
Machine learning is a subset of artificial intelligence.
It focuses on developing algorithms that can learn from data.
"""

SAMPLE_PDF_TEXT = """
Research Paper: Machine Learning Applications

Abstract:
This paper discusses various applications of machine learning
in different domains including healthcare, finance, and robotics.

Introduction:
Machine learning has revolutionized many industries...
"""

SAMPLE_YOUTUBE_TRANSCRIPT = [
    {"text": "Welcome to this video", "start": 0.0, "duration": 2.0},
    {"text": "Today we'll discuss AI", "start": 2.0, "duration": 3.0},
    {"text": "Let's get started", "start": 5.0, "duration": 2.0},
]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_document():
    """Create a sample LangChain Document."""
    return Document(
        page_content=SAMPLE_TEXT,
        metadata={
            "source": "https://example.com/article",
            "source_type": "article",
            "title": "Sample Article"
        }
    )


@pytest.fixture
def sample_documents():
    """Create a list of sample LangChain Documents."""
    return [
        Document(
            page_content="First document about machine learning.",
            metadata={"source": "https://example.com/1", "source_type": "article"}
        ),
        Document(
            page_content="Second document about artificial intelligence.",
            metadata={"source": "https://example.com/2", "source_type": "pdf"}
        ),
        Document(
            page_content="Third document about neural networks.",
            metadata={"source": "https://example.com/3", "source_type": "article"}
        ),
    ]


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = Mock()

    # Mock chat completions
    completion_response = Mock()
    completion_response.choices = [Mock()]
    completion_response.choices[0].message.content = "This is a test response."
    completion_response.usage.total_tokens = 100
    completion_response.usage.prompt_tokens = 50
    completion_response.usage.completion_tokens = 50

    client.chat.completions.create.return_value = completion_response

    # Mock embeddings
    embedding_response = Mock()
    embedding_response.data = [Mock(embedding=[0.1] * 1536)]
    client.embeddings.create.return_value = embedding_response

    return client


@pytest.fixture
def mock_tavily_client():
    """Create a mock Tavily client."""
    client = Mock()

    # Mock search response
    client.search.return_value = {
        "results": [
            {
                "title": "Test Article 1",
                "url": "https://example.com/1",
                "content": "This is test content 1",
                "score": 0.95
            },
            {
                "title": "Test Article 2",
                "url": "https://example.com/2",
                "content": "This is test content 2",
                "score": 0.90
            }
        ]
    }

    return client


@pytest.fixture
def mock_search_results():
    """Create mock SearchResult objects."""
    from src.ingestion.web_search import SearchResult

    return [
        SearchResult(
            title="Machine Learning Basics",
            url="https://arxiv.org/pdf/1234.pdf",
            content_snippet="An introduction to machine learning concepts",
            source_type="pdf",
            score=0.95
        ),
        SearchResult(
            title="AI in Healthcare",
            url="https://example.com/ai-health",
            content_snippet="Applications of AI in medical diagnosis",
            source_type="article",
            score=0.90
        ),
        SearchResult(
            title="Deep Learning Tutorial",
            url="https://www.youtube.com/watch?v=test123",
            content_snippet="Video tutorial on deep learning",
            source_type="video",
            score=0.85
        ),
    ]


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file."""
    file_path = temp_dir / "sample_article.txt"
    file_path.write_text(
        "Source: https://example.com\n"
        "Title: Sample Article\n"
        "=" * 80 + "\n\n"
        + SAMPLE_TEXT
    )
    return file_path


@pytest.fixture
def sample_pdf_file(temp_dir):
    """Create a sample PDF-like file (for testing purposes)."""
    file_path = temp_dir / "sample_paper.pdf"
    file_path.write_text(SAMPLE_PDF_TEXT)
    return file_path


@pytest.fixture
def mock_cost_tracker():
    """Create a mock CostTracker."""
    tracker = Mock()
    tracker.track_openai_call.return_value = 0.001
    tracker.track_embedding_call.return_value = 0.0001
    tracker.track_tavily_search.return_value = 0.01
    tracker.get_session_costs.return_value = {
        "total": 0.05,
        "by_model": {"gpt-4o": 0.03, "text-embedding-3-small": 0.02}
    }
    return tracker


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    import logging
    logging.root.handlers = []
    yield


@pytest.fixture
def mock_youtube_transcript():
    """Create mock YouTube transcript data."""
    return SAMPLE_YOUTUBE_TRANSCRIPT


@pytest.fixture
def disable_network(monkeypatch):
    """Disable network requests for tests."""
    import socket

    def guard(*args, **kwargs):
        raise RuntimeError("Network access not allowed in tests")

    monkeypatch.setattr(socket, "socket", guard)
