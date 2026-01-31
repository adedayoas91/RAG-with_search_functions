"""
Ingestion module for Agentic RAG System.

This package provides functions to search, filter, load, and chunk content
from various sources (web articles, PDFs, YouTube videos).
"""

# Search clients
from .web_search import TavilySearchClient, SearchResult
from .google_search import GoogleSearchClient

# Source filtering and ranking
from .source_filter import (
    filter_sources_by_relevance,
    deduplicate_sources,
    rank_sources_by_type
)

# Source summarization
from .source_summarizer import (
    SourceSummary,
    summarize_source,
    summarize_sources_batch,
    summarize_sources_sync
)

# Article downloading and parsing
from .article_downloader import (
    download_articles_from_sources,
    create_query_directory,
    is_downloadable_article
)

# Content loaders
from .pdf_loader import load_pdf_source, load_pdf_from_file
from .article_loader import load_article
from .text_loader import load_text_file
from .yt_bot import load_youtube_video
from .chunker import chunk_documents

# Local document loading
from .local_document_loader import (
    get_document_source_mode,
    load_local_documents,
    get_local_documents_path,
    print_document_summary
)

__all__ = [
    # Search clients
    "TavilySearchClient",
    "GoogleSearchClient",
    "SearchResult",

    # Source filtering
    "filter_sources_by_relevance",
    "deduplicate_sources",
    "rank_sources_by_type",

    # Source summarization
    "SourceSummary",
    "summarize_source",
    "summarize_sources_batch",
    "summarize_sources_sync",

    # Article downloading
    "download_articles_from_sources",
    "create_query_directory",
    "is_downloadable_article",

    # Content loaders
    "load_pdf_source",
    "load_pdf_from_file",
    "load_article",
    "load_text_file",
    "load_youtube_video",

    # Local document loading
    "get_document_source_mode",
    "load_local_documents",
    "get_local_documents_path",
    "print_document_summary",

    # Chunking
    "chunk_documents",
]
