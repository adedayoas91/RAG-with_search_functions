"""
Unit tests for ChromaDB vector store.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from src.vectorstore.chroma_store import ChromaVectorStore


@pytest.mark.unit
class TestChromaVectorStore:
    """Tests for ChromaDB vector store."""

    @patch('chromadb.PersistentClient')
    def test_initialization(self, mock_chroma_client):
        """Test initializing ChromaDB vector store."""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory="./test_db",
            embedding_function=Mock()
        )

        assert store.collection_name == "test_collection"
        mock_chroma_client.assert_called_once_with(path="./test_db")

    @patch('chromadb.PersistentClient')
    def test_add_documents(self, mock_chroma_client, sample_documents):
        """Test adding documents to vector store."""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_function = Mock()
        mock_embedding_function.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]

        store = ChromaVectorStore(
            collection_name="test",
            persist_directory="./test_db",
            embedding_function=mock_embedding_function
        )

        ids = store.add_documents(sample_documents)

        assert len(ids) == len(sample_documents)
        mock_collection.add.assert_called_once()

    @patch('chromadb.PersistentClient')
    def test_add_empty_documents(self, mock_chroma_client):
        """Test adding empty document list."""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        store = ChromaVectorStore(
            collection_name="test",
            persist_directory="./test_db",
            embedding_function=Mock()
        )

        ids = store.add_documents([])

        assert ids == []
        mock_collection.add.assert_not_called()

    @patch('chromadb.PersistentClient')
    def test_similarity_search(self, mock_chroma_client):
        """Test similarity search."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['First result', 'Second result']],
            'metadatas': [[{'source': 'url1'}, {'source': 'url2'}]],
            'distances': [[0.1, 0.2]]
        }

        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_function = Mock()
        mock_embedding_function.embed_query.return_value = [0.5] * 1536

        store = ChromaVectorStore(
            collection_name="test",
            persist_directory="./test_db",
            embedding_function=mock_embedding_function
        )

        results = store.similarity_search("test query", k=2)

        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        assert results[0].page_content == 'First result'
        assert results[1].page_content == 'Second result'

    @patch('chromadb.PersistentClient')
    def test_similarity_search_with_score(self, mock_chroma_client):
        """Test similarity search with relevance scores."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Result']],
            'metadatas': [[{'source': 'url1'}]],
            'distances': [[0.15]]
        }

        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_function = Mock()
        mock_embedding_function.embed_query.return_value = [0.5] * 1536

        store = ChromaVectorStore(
            collection_name="test",
            persist_directory="./test_db",
            embedding_function=mock_embedding_function
        )

        results = store.similarity_search_with_score("test query", k=1)

        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert score >= 0.0 and score <= 1.0

    @patch('chromadb.PersistentClient')
    def test_get_collection_stats(self, mock_chroma_client):
        """Test getting collection statistics."""
        mock_collection = Mock()
        mock_collection.count.return_value = 42

        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        store = ChromaVectorStore(
            collection_name="test",
            persist_directory="./test_db",
            embedding_function=Mock()
        )

        stats = store.get_collection_stats()

        assert stats["collection_name"] == "test"
        assert stats["num_documents"] == 42

    @patch('chromadb.PersistentClient')
    def test_delete_collection(self, mock_chroma_client):
        """Test deleting collection."""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        store = ChromaVectorStore(
            collection_name="test",
            persist_directory="./test_db",
            embedding_function=Mock()
        )

        store.delete_collection()

        mock_client_instance.delete_collection.assert_called_once_with("test")

    @patch('chromadb.PersistentClient')
    def test_add_documents_with_metadata(self, mock_chroma_client):
        """Test that metadata is preserved when adding documents."""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_function = Mock()
        mock_embedding_function.embed_documents.return_value = [[0.1] * 1536]

        store = ChromaVectorStore(
            collection_name="test",
            persist_directory="./test_db",
            embedding_function=mock_embedding_function
        )

        doc = Document(
            page_content="Test content",
            metadata={"source": "test.pdf", "page": 1, "custom_field": "value"}
        )

        store.add_documents([doc])

        # Verify metadata was passed to add
        call_args = mock_collection.add.call_args
        assert call_args is not None
        assert 'metadatas' in call_args[1]
        assert call_args[1]['metadatas'][0]['source'] == 'test.pdf'
        assert call_args[1]['metadatas'][0]['page'] == 1

    @patch('chromadb.PersistentClient')
    def test_search_with_filter(self, mock_chroma_client):
        """Test similarity search with metadata filter."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Filtered result']],
            'metadatas': [[{'source': 'target.pdf'}]],
            'distances': [[0.1]]
        }

        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_function = Mock()
        mock_embedding_function.embed_query.return_value = [0.5] * 1536

        store = ChromaVectorStore(
            collection_name="test",
            persist_directory="./test_db",
            embedding_function=mock_embedding_function
        )

        results = store.similarity_search(
            "test query",
            k=1,
            filter={"source": "target.pdf"}
        )

        assert len(results) == 1
        mock_collection.query.assert_called_once()
