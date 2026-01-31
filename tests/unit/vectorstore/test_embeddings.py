"""
Unit tests for embedding functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.vectorstore.embeddings import OpenAIEmbedding


@pytest.mark.unit
class TestOpenAIEmbedding:
    """Tests for OpenAI embedding wrapper."""

    def test_initialization(self, mock_openai_client):
        """Test initializing OpenAI embedding model."""
        embedder = OpenAIEmbedding(
            api_key="test-key",
            model="text-embedding-3-small"
        )

        assert embedder.model == "text-embedding-3-small"
        assert embedder.dimension == 1536

    def test_initialization_with_cost_tracker(self, mock_openai_client, mock_cost_tracker):
        """Test initialization with cost tracker."""
        embedder = OpenAIEmbedding(
            api_key="test-key",
            cost_tracker=mock_cost_tracker
        )

        assert embedder.cost_tracker == mock_cost_tracker

    def test_embed_documents(self, mock_openai_client):
        """Test embedding multiple documents."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_embedding1 = Mock()
        mock_embedding1.embedding = [0.1] * 1536
        mock_embedding2 = Mock()
        mock_embedding2.embedding = [0.2] * 1536
        mock_response.data = [mock_embedding1, mock_embedding2]
        mock_openai_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedding(api_key="test-key")
        embedder.client = mock_openai_client

        texts = ["First document", "Second document"]
        embeddings = embedder.embed_documents(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536
        assert embeddings[0][0] == 0.1
        assert embeddings[1][0] == 0.2

    def test_embed_query(self, mock_openai_client):
        """Test embedding a single query."""
        mock_response = Mock()
        mock_embedding = Mock()
        mock_embedding.embedding = [0.5] * 1536
        mock_response.data = [mock_embedding]
        mock_openai_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedding(api_key="test-key")
        embedder.client = mock_openai_client

        query_embedding = embedder.embed_query("What is machine learning?")

        assert len(query_embedding) == 1536
        assert query_embedding[0] == 0.5

    def test_embed_empty_list(self, mock_openai_client):
        """Test embedding empty document list."""
        embedder = OpenAIEmbedding(api_key="test-key")
        embedder.client = mock_openai_client

        embeddings = embedder.embed_documents([])

        assert embeddings == []

    def test_cost_tracking_on_embed(self, mock_openai_client, mock_cost_tracker):
        """Test that cost tracking works during embedding."""
        mock_response = Mock()
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response.data = [mock_embedding]
        mock_response.usage = Mock(total_tokens=100)
        mock_openai_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedding(api_key="test-key", cost_tracker=mock_cost_tracker)
        embedder.client = mock_openai_client

        embedder.embed_documents(["test"])

        mock_cost_tracker.track_openai_call.assert_called_once()

    def test_batch_processing(self, mock_openai_client):
        """Test that large batches are processed correctly."""
        # Create 100 mock embeddings
        mock_embeddings = []
        for i in range(100):
            mock_emb = Mock()
            mock_emb.embedding = [i * 0.01] * 1536
            mock_embeddings.append(mock_emb)

        mock_response = Mock()
        mock_response.data = mock_embeddings
        mock_openai_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedding(api_key="test-key")
        embedder.client = mock_openai_client

        texts = [f"Document {i}" for i in range(100)]
        embeddings = embedder.embed_documents(texts)

        assert len(embeddings) == 100
        assert all(len(emb) == 1536 for emb in embeddings)

    def test_different_model_dimensions(self, mock_openai_client):
        """Test using different embedding models with different dimensions."""
        embedder_large = OpenAIEmbedding(
            api_key="test-key",
            model="text-embedding-3-large"
        )

        assert embedder_large.dimension == 3072

    def test_embed_with_special_characters(self, mock_openai_client):
        """Test embedding text with special characters."""
        mock_response = Mock()
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response.data = [mock_embedding]
        mock_openai_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedding(api_key="test-key")
        embedder.client = mock_openai_client

        text_with_special_chars = "Text with émojis 🚀 and symbols: €, ñ, 中文"
        embedding = embedder.embed_query(text_with_special_chars)

        assert len(embedding) == 1536

    def test_api_error_handling(self, mock_openai_client):
        """Test handling of API errors."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")

        embedder = OpenAIEmbedding(api_key="test-key")
        embedder.client = mock_openai_client

        with pytest.raises(Exception):
            embedder.embed_documents(["test"])
