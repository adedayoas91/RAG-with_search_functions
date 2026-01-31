"""
ChromaDB vector store for document embeddings.

This module provides a wrapper around ChromaDB for storing and retrieving
document embeddings.
"""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from src.vectorstore.embeddings import EmbeddingModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """ChromaDB vector store for RAG system."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_model: EmbeddingModel
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            embedding_model: Embedding model instance

        Example:
            >>> from src.vectorstore.embeddings import OpenAIEmbedding
            >>> embedder = OpenAIEmbedding(api_key="your-key")
            >>> store = ChromaVectorStore("./data/chroma_db", "my_collection", embedder)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB at: {persist_directory}")

        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

            logger.info(
                f"ChromaDB initialized: collection='{collection_name}', "
                f"documents={self.collection.count()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects to add

        Example:
            >>> docs = [Document(page_content="text", metadata={"source": "url"})]
            >>> store.add_documents(docs)
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents to vector store")

        try:
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Generate embeddings
            logger.debug(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embedding_model.embed_documents(texts)

            # Generate IDs
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(documents))]

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            logger.info(
                f"Successfully added {len(documents)} documents. "
                f"Total in collection: {self.collection.count()}"
            )

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter (e.g., {"source_type": "pdf"})

        Returns:
            List of tuples (Document, similarity_score)

        Example:
            >>> results = store.similarity_search("AI safety", k=5)
            >>> for doc, score in results:
            ...     print(f"Score: {score:.3f} - {doc.page_content[:100]}")
        """
        logger.debug(f"Searching for: '{query}' (k={k})")

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter  # Optional metadata filter
            )

            # Parse results
            documents_with_scores = []

            if results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    doc = Document(
                        page_content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i]
                    )
                    # ChromaDB returns distances, convert to similarity (1 - distance)
                    distance = results['distances'][0][i]
                    similarity = 1.0 - distance

                    documents_with_scores.append((doc, similarity))

            logger.info(f"Found {len(documents_with_scores)} similar documents")

            return documents_with_scores

        except Exception as e:
            logger.error(f"Failed to search: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()

            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "persist_directory": str(self.persist_directory)
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "total_documents": 0,
                "persist_directory": str(self.persist_directory),
                "error": str(e)
            }

    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.

        Warning: This operation cannot be undone.
        """
        logger.warning(f"Clearing collection: {self.collection_name}")

        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)

            # Recreate empty collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"Collection cleared: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            raise
