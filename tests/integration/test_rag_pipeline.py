"""
Integration tests for end-to-end RAG pipeline.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from langchain_core.documents import Document


@pytest.mark.integration
class TestRAGPipeline:
    """Integration tests for complete RAG workflow."""

    def test_document_processing_pipeline(self, sample_documents, temp_dir):
        """Test complete document processing: load -> chunk -> embed."""
        from src.ingestion.chunker import chunk_documents

        # Chunk documents
        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
        assert all(hasattr(c, 'page_content') for c in chunks)
        assert all(hasattr(c, 'metadata') for c in chunks)

    @patch('src.ingestion.article_downloader.download_article_wget')
    @patch('src.ingestion.article_downloader.parse_and_save_article')
    def test_download_and_load_pipeline(
        self, mock_parse, mock_download, temp_dir, mock_search_results
    ):
        """Test download/parse -> load pipeline."""
        from src.ingestion.article_downloader import download_articles_from_sources
        from src.ingestion.text_loader import load_text_file

        # Mock successful parse
        test_file = temp_dir / "test_article.txt"
        test_file.write_text("Source: https://example.com\nTitle: Test\n\nContent here")
        mock_parse.return_value = (True, test_file)
        mock_download.return_value = (False, None)

        # Download/parse articles
        saved = download_articles_from_sources(
            sources=mock_search_results[:1],
            query="test",
            base_dir=str(temp_dir),
            min_downloads=1,
            max_workers=1
        )

        assert len(saved) >= 1

        # Load the saved file
        _, file_path = saved[0]
        doc = load_text_file(str(file_path), "https://example.com")

        assert doc is not None
        assert len(doc.page_content) > 0

    def test_chunk_and_embed_pipeline(self, mock_openai_client, sample_documents):
        """Test chunking and embedding pipeline."""
        from src.ingestion.chunker import chunk_documents
        from src.vectorstore.embeddings import OpenAIEmbedding

        # Chunk
        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)

        # Mock embeddings
        embedder = OpenAIEmbedding(
            api_key="test-key",
            model="text-embedding-3-small"
        )
        embedder.client = mock_openai_client

        # Embed a few chunks
        texts = [c.page_content for c in chunks[:3]]
        embeddings = embedder.embed_documents(texts)

        assert len(embeddings) == len(texts)
        assert all(isinstance(e, list) for e in embeddings)

    def test_retrieve_and_generate_pipeline(
        self, mock_openai_client, sample_documents
    ):
        """Test retrieval and answer generation pipeline."""
        from src.generation.answer_generator import RAGAnswerGenerator

        # Use sample documents as "retrieved" context
        generator = RAGAnswerGenerator(client=mock_openai_client)

        answer = generator.generate_answer(
            query="What is machine learning?",
            context_documents=sample_documents,
            temperature=0.2,
            max_tokens=500
        )

        assert answer is not None
        assert len(answer.answer) > 0
        assert len(answer.sources) > 0
        assert "## Sources" in answer.answer

    @pytest.mark.slow
    def test_full_pipeline_mock(
        self, mock_openai_client, mock_tavily_client, temp_dir
    ):
        """Test full pipeline with mocked external services."""
        from src.ingestion.web_search import SearchResult
        from src.ingestion.chunker import chunk_documents
        from src.generation.answer_generator import RAGAnswerGenerator

        # 1. Simulate search results
        search_results = [
            SearchResult(
                title="ML Article",
                url="https://example.com/ml",
                content_snippet="Machine learning overview",
                source_type="article",
                score=0.9
            )
        ]

        # 2. Create mock documents
        documents = [
            Document(
                page_content="Machine learning is a field of AI.",
                metadata={"source": "https://example.com/ml", "source_type": "article"}
            )
        ]

        # 3. Chunk documents
        chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 0

        # 4. Generate answer
        generator = RAGAnswerGenerator(client=mock_openai_client)
        answer = generator.generate_answer(
            query="What is machine learning?",
            context_documents=chunks
        )

        assert answer is not None
        assert len(answer.answer) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestParallelProcessing:
    """Integration tests for parallel processing features."""

    def test_parallel_chunking(self, sample_documents):
        """Test parallel document chunking with Ray."""
        from src.vectorstore.parallel_embedding import parallel_chunk_documents

        chunks = parallel_chunk_documents(
            documents=sample_documents,
            chunk_size=100,
            chunk_overlap=20,
            num_workers=2
        )

        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_parallel_downloads(self, temp_dir):
        """Test parallel article downloading."""
        from src.ingestion.article_downloader import download_articles_from_sources
        from src.ingestion.web_search import SearchResult

        # Create mock sources
        sources = [
            SearchResult(
                title=f"Article {i}",
                url=f"https://example.com/{i}",
                content_snippet=f"Content {i}",
                source_type="article",
                score=0.9
            )
            for i in range(5)
        ]

        # This will try to parse these (and likely fail), but tests parallel execution
        with patch('src.ingestion.article_downloader.parse_and_save_article') as mock_parse:
            test_file = temp_dir / "test.txt"
            test_file.write_text("Test content")
            mock_parse.return_value = (True, test_file)

            saved = download_articles_from_sources(
                sources=sources,
                query="test",
                base_dir=str(temp_dir),
                min_downloads=3,
                max_workers=3
            )

            # Should have attempted parallel processing
            assert len(saved) >= 3


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Integration tests for vector store operations."""

    @patch('chromadb.PersistentClient')
    def test_add_and_search_pipeline(self, mock_chroma_client, sample_documents):
        """Test adding documents and searching vector store."""
        from src.vectorstore.chroma_store import ChromaVectorStore
        from src.vectorstore.embeddings import OpenAIEmbedding
        from src.ingestion.chunker import chunk_documents

        # Mock ChromaDB
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [[sample_documents[0].page_content, sample_documents[1].page_content]],
            'metadatas': [[sample_documents[0].metadata, sample_documents[1].metadata]],
            'distances': [[0.1, 0.2]]
        }
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        # Mock embeddings
        mock_embedding_func = Mock()
        mock_embedding_func.embed_documents.return_value = [[0.1] * 1536 for _ in sample_documents]
        mock_embedding_func.embed_query.return_value = [0.5] * 1536

        # Chunk documents
        chunks = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)

        # Create vector store and add documents
        store = ChromaVectorStore(
            collection_name="test",
            persist_directory="./test_db",
            embedding_function=mock_embedding_func
        )
        store.add_documents(chunks)

        # Search
        results = store.similarity_search("test query", k=2)

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)

    @patch('chromadb.PersistentClient')
    def test_cost_tracking_integration(self, mock_chroma_client, mock_openai_client):
        """Test cost tracking throughout pipeline."""
        from src.utils.cost_tracker import CostTracker
        from src.generation.answer_generator import RAGAnswerGenerator
        from src.vectorstore.embeddings import OpenAIEmbedding

        # Create cost tracker
        tracker = CostTracker()

        # Mock embedding calls
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)],
            usage=Mock(total_tokens=100)
        )

        # Track embedding cost
        embedder = OpenAIEmbedding(api_key="test-key", cost_tracker=tracker)
        embedder.client = mock_openai_client
        embedder.embed_documents(["test"])

        # Track generation cost
        generator = RAGAnswerGenerator(client=mock_openai_client, cost_tracker=tracker)
        generator.generate_answer(
            query="test",
            context_documents=[Document(page_content="test", metadata={"source": "test"})]
        )

        # Verify costs were tracked
        assert tracker.total_cost > 0
        assert len(tracker.call_history) >= 2


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling across pipeline."""

    def test_empty_document_pipeline(self):
        """Test pipeline with empty documents."""
        from src.ingestion.chunker import chunk_documents

        empty_docs = []
        chunks = chunk_documents(empty_docs, chunk_size=100, chunk_overlap=20)

        assert chunks == []

    @patch('src.ingestion.article_downloader.parse_and_save_article')
    def test_failed_downloads_handling(self, mock_parse, temp_dir, mock_search_results):
        """Test handling of failed downloads."""
        from src.ingestion.article_downloader import download_articles_from_sources

        # All downloads fail
        mock_parse.return_value = (False, None)

        with patch('builtins.input', return_value='n'):
            with pytest.raises(SystemExit):
                download_articles_from_sources(
                    sources=mock_search_results[:3],
                    query="test",
                    base_dir=str(temp_dir),
                    min_downloads=10,
                    max_workers=1
                )

    def test_empty_context_generation(self, mock_openai_client):
        """Test answer generation with empty context."""
        from src.generation.answer_generator import RAGAnswerGenerator

        generator = RAGAnswerGenerator(client=mock_openai_client)
        answer = generator.generate_answer(
            query="test query",
            context_documents=[]
        )

        assert answer is not None
        assert len(answer.sources) == 0

    def test_invalid_chunk_size(self):
        """Test chunking with invalid parameters."""
        from src.ingestion.chunker import chunk_documents

        doc = Document(page_content="Short text", metadata={"source": "test"})

        # Chunk size larger than document should still work
        chunks = chunk_documents([doc], chunk_size=10000, chunk_overlap=100)
        assert len(chunks) >= 1


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndScenarios:
    """End-to-end integration test scenarios."""

    @patch('src.ingestion.article_downloader.download_article_wget')
    @patch('chromadb.PersistentClient')
    def test_pdf_download_to_answer(
        self, mock_chroma_client, mock_download, mock_openai_client, temp_dir
    ):
        """Test complete flow: PDF download -> chunk -> embed -> search -> answer."""
        from src.ingestion.pdf_loader import load_pdf_from_file
        from src.ingestion.chunker import chunk_documents
        from src.vectorstore.chroma_store import ChromaVectorStore
        from src.generation.answer_generator import RAGAnswerGenerator

        # 1. Mock PDF download
        test_pdf = temp_dir / "paper.pdf"
        test_pdf.write_bytes(b"test")
        mock_download.return_value = (True, test_pdf)

        # 2. Mock PDF extraction
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {"title": "Test Paper"}
            mock_doc.__len__.return_value = 1
            mock_page = Mock()
            mock_page.get_text.return_value = "Machine learning research content"
            mock_doc.__getitem__.return_value = mock_page
            mock_fitz.return_value = mock_doc

            doc = load_pdf_from_file(str(test_pdf))

        # 3. Chunk
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)

        # 4. Mock vector store
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [[chunks[0].page_content]],
            'metadatas': [[chunks[0].metadata]],
            'distances': [[0.1]]
        }
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_embedding_func = Mock()
        mock_embedding_func.embed_documents.return_value = [[0.1] * 1536]
        mock_embedding_func.embed_query.return_value = [0.5] * 1536

        store = ChromaVectorStore(
            collection_name="test",
            persist_directory=str(temp_dir / "chroma"),
            embedding_function=mock_embedding_func
        )
        store.add_documents(chunks)

        # 5. Search
        retrieved = store.similarity_search("machine learning", k=3)

        # 6. Generate answer
        generator = RAGAnswerGenerator(client=mock_openai_client)
        answer = generator.generate_answer(
            query="What is machine learning?",
            context_documents=retrieved
        )

        # Verify complete pipeline
        assert len(chunks) > 0
        assert len(retrieved) > 0
        assert answer is not None
        assert len(answer.answer) > 0

    @patch('requests.get')
    @patch('chromadb.PersistentClient')
    def test_article_parsing_to_answer(
        self, mock_chroma_client, mock_requests, mock_openai_client, temp_dir
    ):
        """Test complete flow: HTML article -> parse -> chunk -> embed -> answer."""
        from src.ingestion.article_downloader import parse_and_save_article
        from src.ingestion.text_loader import load_text_file
        from src.ingestion.chunker import chunk_documents
        from src.generation.answer_generator import RAGAnswerGenerator

        # 1. Mock article HTML
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <body>
                <article>
                    <h1>AI Research</h1>
                    <p>Deep learning has revolutionized computer vision.</p>
                </article>
            </body>
        </html>
        """
        mock_requests.return_value = mock_response

        # 2. Parse article
        success, file_path = parse_and_save_article(
            url="https://example.com/ai-article",
            output_dir=temp_dir,
            title="AI Research"
        )
        assert success is True

        # 3. Load parsed article
        doc = load_text_file(str(file_path), "https://example.com/ai-article")

        # 4. Chunk
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)

        # 5. Generate answer
        generator = RAGAnswerGenerator(client=mock_openai_client)
        answer = generator.generate_answer(
            query="What is deep learning?",
            context_documents=chunks
        )

        # Verify complete pipeline
        assert doc is not None
        assert "Deep learning" in doc.page_content
        assert len(chunks) > 0
        assert answer is not None
        assert "## Sources" in answer.answer
