"""
Unit tests for ChromaDB vector store.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from src.vectorstore.chroma_store import ChromaVectorStore


@pytest.mark.unit
class TestChromaVectorStore:
    """Tests for ChromaDB vector store."""

    @patch('chromadb.PersistentClient')
    def test_initialization(self, mock_chroma_client, temp_dir):
        """Test initializing ChromaDB vector store."""
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_model = Mock()

        store = ChromaVectorStore(
            persist_directory=str(temp_dir),
            collection_name="test_collection",
            embedding_model=mock_embedding_model
        )

        assert store.collection_name == "test_collection"
        mock_chroma_client.assert_called_once()

    @patch('chromadb.PersistentClient')
    def test_add_documents(self, mock_chroma_client, sample_documents, temp_dir):
        """Test adding documents to vector store."""
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_model = Mock()
        mock_embedding_model.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]

        store = ChromaVectorStore(
            persist_directory=str(temp_dir),
            collection_name="test",
            embedding_model=mock_embedding_model
        )

        store.add_documents(sample_documents)

        mock_collection.add.assert_called_once()

    @patch('chromadb.PersistentClient')
    def test_add_empty_documents(self, mock_chroma_client, temp_dir):
        """Test adding empty document list."""
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_model = Mock()

        store = ChromaVectorStore(
            persist_directory=str(temp_dir),
            collection_name="test",
            embedding_model=mock_embedding_model
        )

        store.add_documents([])

        mock_collection.add.assert_not_called()

    @patch('chromadb.PersistentClient')
    def test_similarity_search(self, mock_chroma_client, temp_dir):
        """Test similarity search."""
        mock_collection = Mock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['First result', 'Second result']],
            'metadatas': [[{'source': 'url1'}, {'source': 'url2'}]],
            'distances': [[0.1, 0.2]]
        }

        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.5] * 1536

        store = ChromaVectorStore(
            persist_directory=str(temp_dir),
            collection_name="test",
            embedding_model=mock_embedding_model
        )

        results = store.similarity_search("test query", k=2)

        assert len(results) == 2
        # Results are (Document, score) tuples
        assert all(isinstance(result, tuple) for result in results)
        assert all(isinstance(result[0], Document) for result in results)
        assert all(isinstance(result[1], float) for result in results)

    @patch('chromadb.PersistentClient')
    def test_similarity_search_with_score(self, mock_chroma_client, temp_dir):
        """Test similarity search with relevance scores."""
        mock_collection = Mock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Result']],
            'metadatas': [[{'source': 'url1'}]],
            'distances': [[0.15]]
        }

        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.5] * 1536

        store = ChromaVectorStore(
            persist_directory=str(temp_dir),
            collection_name="test",
            embedding_model=mock_embedding_model
        )

        results = store.similarity_search("test query", k=1)

        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert score >= 0.0 and score <= 1.0

    @patch('chromadb.PersistentClient')
    def test_get_collection_stats(self, mock_chroma_client, temp_dir):
        """Test getting collection statistics."""
        mock_collection = Mock()
        mock_collection.count.return_value = 42

        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_model = Mock()

        store = ChromaVectorStore(
            persist_directory=str(temp_dir),
            collection_name="test",
            embedding_model=mock_embedding_model
        )

        stats = store.get_collection_stats()

        assert stats["collection_name"] == "test"
        assert stats["total_documents"] == 42

    @patch('chromadb.PersistentClient')
    def test_clear_collection(self, mock_chroma_client, temp_dir):
        """Test clearing collection."""
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_model = Mock()

        store = ChromaVectorStore(
            persist_directory=str(temp_dir),
            collection_name="test",
            embedding_model=mock_embedding_model
        )

        store.clear_collection()

        # Should delete and recreate
        mock_client_instance.delete_collection.assert_called_once_with(name="test")
        assert mock_client_instance.get_or_create_collection.call_count >= 2  # Initial + recreate

    @patch('chromadb.PersistentClient')
    def test_add_documents_with_metadata(self, mock_chroma_client, temp_dir):
        """Test that metadata is preserved when adding documents."""
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_model = Mock()
        mock_embedding_model.embed_documents.return_value = [[0.1] * 1536]

        store = ChromaVectorStore(
            persist_directory=str(temp_dir),
            collection_name="test",
            embedding_model=mock_embedding_model
        )

        doc = Document(
            page_content="Test content",
            metadata={"source": "test.pdf", "page": 1, "custom_field": "value"}
        )

        store.add_documents([doc])

        # Verify add was called
        assert mock_collection.add.called

    @patch('chromadb.PersistentClient')
    def test_search_with_filter(self, mock_chroma_client, temp_dir):
        """Test similarity search with metadata filter."""
        mock_collection = Mock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Filtered result']],
            'metadatas': [[{'source': 'target.pdf'}]],
            'distances': [[0.1]]
        }

        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_model = Mock()
        mock_embedding_model.embed_query.return_value = [0.5] * 1536

        store = ChromaVectorStore(
            persist_directory=str(temp_dir),
            collection_name="test",
            embedding_model=mock_embedding_model
        )

        results = store.similarity_search(
            "test query",
            k=1,
            filter={"source": "target.pdf"}
        )

        assert len(results) == 1
        mock_collection.query.assert_called_once()
