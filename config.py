"""
Configuration management for Agentic RAG System.

This module provides centralized configuration using Pydantic for validation.
All settings can be overridden via environment variables.
"""

import os
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class APIConfig(BaseModel):
    """API keys and endpoints configuration."""

    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key"
    )
    tavily_api_key: str = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", ""),
        description="Tavily API key for web search"
    )


class ModelConfig(BaseModel):
    """LLM and embedding model configuration."""

    # OpenAI models
    summarization_model: str = Field(
        default="gpt-4o-mini",
        description="Model for source summarization"
    )
    generation_model: str = Field(
        default="gpt-4o",
        description="Model for answer generation"
    )
    multi_query_model: str = Field(
        default="gpt-4o-mini",
        description="Model for query expansion"
    )

    # Model parameters
    temperature: float = Field(
        default=0.2,
        description="Temperature for generation (0.0-2.0)"
    )
    max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for generation"
    )

    # Embedding configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers embedding model"
    )
    embedding_type: str = Field(
        default="sentence-transformers",
        description="Type: 'sentence-transformers' or 'openai'"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model if using OpenAI"
    )


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    persist_directory: Path = Field(
        default=Path("./data/chroma_db"),
        description="ChromaDB persistence directory"
    )
    collection_name: str = Field(
        default="agentic_rag_v1",
        description="ChromaDB collection name"
    )
    similarity_metric: str = Field(
        default="cosine",
        description="Similarity metric for vector search"
    )


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""

    chunk_size: int = Field(
        default=80,
        description="Size of text chunks in characters"
    )
    chunk_overlap: int = Field(
        default=20,
        description="Overlap between consecutive chunks"
    )
    separators: List[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Separators for recursive text splitting"
    )


class SearchConfig(BaseModel):
    """Web search configuration."""

    max_search_results: int = Field(
        default=10,
        description="Maximum number of search results to fetch"
    )
    relevance_threshold: float = Field(
        default=0.7,
        description="Minimum relevance score for source filtering (0.0-1.0)"
    )
    content_types: List[str] = Field(
        default=["article", "pdf", "video"],
        description="Content types to search for"
    )
    search_depth: str = Field(
        default="advanced",
        description="Tavily search depth: 'basic' or 'advanced'"
    )


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    retrieval_k: int = Field(
        default=5,
        description="Number of documents to retrieve"
    )
    use_query_expansion: bool = Field(
        default=True,
        description="Whether to use multi-query expansion"
    )
    num_expanded_queries: int = Field(
        default=3,
        description="Number of queries to generate for expansion"
    )
    max_context_tokens: int = Field(
        default=8000,
        description="Maximum tokens for retrieved context"
    )


class CostConfig(BaseModel):
    """API cost pricing configuration (USD per 1K tokens)."""

    # OpenAI pricing
    gpt_4o_mini_input: float = Field(
        default=0.00015,
        description="GPT-4o-mini input cost per 1K tokens"
    )
    gpt_4o_mini_output: float = Field(
        default=0.0006,
        description="GPT-4o-mini output cost per 1K tokens"
    )
    gpt_4o_input: float = Field(
        default=0.0025,
        description="GPT-4o input cost per 1K tokens"
    )
    gpt_4o_output: float = Field(
        default=0.01,
        description="GPT-4o output cost per 1K tokens"
    )

    # Embedding pricing
    text_embedding_3_small: float = Field(
        default=0.00002,
        description="OpenAI text-embedding-3-small cost per 1K tokens"
    )
    sentence_transformers_cost: float = Field(
        default=0.0,
        description="Sentence-transformers cost (free, local)"
    )

    # Tavily pricing
    tavily_basic_search: float = Field(
        default=0.005,
        description="Tavily basic search cost per query"
    )
    tavily_advanced_search: float = Field(
        default=0.01,
        description="Tavily advanced search cost per query"
    )


class PathsConfig(BaseModel):
    """File paths configuration."""

    data_dir: Path = Field(
        default=Path("./data"),
        description="Data directory path"
    )
    analytics_file: Path = Field(
        default=Path("./data/analytics.json"),
        description="Analytics data file"
    )
    cost_log_file: Path = Field(
        default=Path("./data/cost_log.json"),
        description="Cost tracking log file"
    )
    log_file: Path = Field(
        default=Path("./data/agentic_rag.log"),
        description="Application log file"
    )


class Config(BaseModel):
    """Main configuration class combining all config sections."""

    api: APIConfig = Field(default_factory=APIConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    def validate_api_keys(self) -> tuple[bool, List[str]]:
        """
        Validate that required API keys are set.

        Returns:
            Tuple of (is_valid, list of missing keys)
        """
        missing_keys = []

        if not self.api.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.api.tavily_api_key:
            missing_keys.append("TAVILY_API_KEY")

        return len(missing_keys) == 0, missing_keys

    def ensure_directories_exist(self) -> None:
        """Create necessary directories if they don't exist."""
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.persist_directory.mkdir(parents=True, exist_ok=True)

    def get_model_cost(self, model: str, token_type: str) -> float:
        """
        Get cost per 1K tokens for a specific model and token type.

        Args:
            model: Model name (e.g., 'gpt-4o-mini')
            token_type: 'input' or 'output'

        Returns:
            Cost per 1K tokens
        """
        if model == "gpt-4o-mini":
            return (
                self.cost.gpt_4o_mini_input if token_type == "input"
                else self.cost.gpt_4o_mini_output
            )
        elif model == "gpt-4o":
            return (
                self.cost.gpt_4o_input if token_type == "input"
                else self.cost.gpt_4o_output
            )
        elif model in ["text-embedding-3-small", "text-embedding-ada-002"]:
            return self.cost.text_embedding_3_small
        elif "sentence-transformers" in model or model == self.model.embedding_model:
            return self.cost.sentence_transformers_cost
        else:
            return 0.0


# Global configuration instance
config = Config()

# Ensure directories exist on import
config.ensure_directories_exist()


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance
    """
    return config


def reload_config() -> Config:
    """
    Reload configuration from environment variables.

    Returns:
        New Config instance
    """
    global config
    load_dotenv(override=True)
    config = Config()
    config.ensure_directories_exist()
    return config
