"""
Web search integration using Tavily API.

This module provides functions to search for relevant sources (articles, PDFs, videos)
using the Tavily search API.
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from config import get_config
from src.utils.logging_config import get_logger
from src.utils.cost_tracker import CostTracker

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""

    title: str
    url: str
    content_snippet: str
    source_type: str  # "article", "pdf", "video"
    score: float
    published_date: Optional[str] = None

    def __str__(self) -> str:
        return f"SearchResult(title='{self.title[:50]}...', type={self.source_type}, score={self.score:.2f})"


class TavilySearchClient:
    """Client for searching web content using Tavily API."""

    def __init__(self, api_key: Optional[str] = None, cost_tracker: Optional[CostTracker] = None):
        """
        Initialize Tavily search client.

        Args:
            api_key: Tavily API key (if None, loads from config)
            cost_tracker: CostTracker instance for logging costs

        Raises:
            ImportError: If tavily-python package is not installed
            ValueError: If API key is not provided or found in config
        """
        if TavilyClient is None:
            raise ImportError(
                "tavily-python package is required. Install with: uv pip install tavily-python"
            )

        self.config = get_config()
        self.api_key = api_key or self.config.api.tavily_api_key

        if not self.api_key:
            raise ValueError(
                "Tavily API key not found. Set TAVILY_API_KEY in .env file"
            )

        self.client = TavilyClient(api_key=self.api_key)
        self.cost_tracker = cost_tracker

        logger.info("Tavily search client initialized")

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search for relevant sources using Tavily API.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_depth: 'basic' or 'advanced' (advanced is more thorough but costs more)
            include_domains: List of domains to include (e.g., ['arxiv.org', 'github.com'])
            exclude_domains: List of domains to exclude

        Returns:
            List of SearchResult objects

        Example:
            >>> client = TavilySearchClient()
            >>> results = client.search("AI safety frameworks", max_results=5)
            >>> for result in results:
            ...     print(f"{result.title} ({result.source_type})")
        """
        logger.info(f"Searching Tavily: '{query}' (max_results={max_results}, depth={search_depth})")

        try:
            # Perform search
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                include_answer=False,  # We don't need the AI-generated answer
                include_raw_content=False  # We'll fetch full content separately
            )

            # Track cost
            if self.cost_tracker:
                self.cost_tracker.track_tavily_search(
                    search_depth=search_depth,
                    num_results=max_results,
                    metadata={"query": query}
                )

            # Parse results
            results = []
            for item in response.get('results', []):
                # Detect source type
                source_type = self._detect_source_type(item.get('url', ''))

                result = SearchResult(
                    title=item.get('title', 'Untitled'),
                    url=item.get('url', ''),
                    content_snippet=item.get('content', ''),
                    source_type=source_type,
                    score=item.get('score', 0.0),
                    published_date=item.get('published_date')
                )
                results.append(result)

            logger.info(f"Found {len(results)} results for query: '{query}'")

            return results

        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            raise

    def _detect_source_type(self, url: str) -> str:
        """
        Detect source type from URL.

        Args:
            url: Source URL

        Returns:
            Source type: 'video', 'pdf', or 'article'
        """
        url_lower = url.lower()

        # Check for YouTube/video
        video_patterns = [
            'youtube.com',
            'youtu.be',
            'vimeo.com',
            'dailymotion.com'
        ]
        if any(pattern in url_lower for pattern in video_patterns):
            return 'video'

        # Check for PDF
        if url_lower.endswith('.pdf') or '/pdf/' in url_lower:
            return 'pdf'

        # Default to article
        return 'article'

    def search_with_filters(
        self,
        query: str,
        content_types: List[str],
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Search with content type filters.

        Args:
            query: Search query
            content_types: List of content types to include: ['article', 'pdf', 'video']
            max_results: Maximum results per content type

        Returns:
            List of SearchResult objects, filtered by content type
        """
        all_results = []

        # For videos, modify query to include video-specific terms
        if 'video' in content_types:
            video_query = f"{query} site:youtube.com OR site:vimeo.com"
            video_results = self.search(video_query, max_results=max_results // len(content_types))
            all_results.extend([r for r in video_results if r.source_type == 'video'])

        # For PDFs, modify query to include PDF-specific terms
        if 'pdf' in content_types:
            pdf_query = f"{query} filetype:pdf"
            pdf_results = self.search(pdf_query, max_results=max_results // len(content_types))
            all_results.extend([r for r in pdf_results if r.source_type == 'pdf'])

        # For articles, search without special filters
        if 'article' in content_types:
            article_results = self.search(query, max_results=max_results)
            all_results.extend([r for r in article_results if r.source_type == 'article'])

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Sort by score (descending)
        unique_results.sort(key=lambda x: x.score, reverse=True)

        # Limit to max_results
        return unique_results[:max_results]
