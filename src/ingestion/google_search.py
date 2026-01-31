"""
Google Search for web sources.

This module provides Google Search integration for finding articles,
PDFs, and academic papers.
"""

from typing import List, Optional
from googlesearch import search as google_search
import time
import requests
from urllib.parse import urlparse

from src.ingestion.web_search import SearchResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def detect_source_type(url: str) -> str:
    """
    Detect source type from URL.

    Args:
        url: URL to check

    Returns:
        Source type: "pdf", "video", or "article"
    """
    url_lower = url.lower()

    # Check for PDF
    if url_lower.endswith('.pdf') or '/pdf/' in url_lower:
        return "pdf"

    # Check for video
    if 'youtube.com' in url_lower or 'youtu.be' in url_lower or 'vimeo.com' in url_lower:
        return "video"

    # Default to article
    return "article"


def is_paywall_domain(url: str) -> bool:
    """
    Check if URL is from a known paywalled domain.

    Args:
        url: URL to check

    Returns:
        True if likely behind paywall
    """
    paywall_indicators = [
        'wsj.com',
        'ft.com',
        'bloomberg.com',
        'economist.com',
        'nytimes.com',
        'washingtonpost.com',
        'newyorker.com',
        'theatlantic.com',
        'subscribe',
        'membership',
        'premium',
    ]

    url_lower = url.lower()
    parsed = urlparse(url_lower)
    domain = parsed.netloc

    for indicator in paywall_indicators:
        if indicator in domain or indicator in url_lower:
            logger.debug(f"Potential paywall detected: {url}")
            return True

    return False


def check_url_accessible(url: str, timeout: int = 5) -> bool:
    """
    Check if URL is accessible (not 403, 401, or paywall).

    Args:
        url: URL to check
        timeout: Request timeout

    Returns:
        True if accessible
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)

        # Check status code
        if response.status_code in [401, 403, 402]:  # 402 is payment required
            logger.debug(f"Access denied for {url}: {response.status_code}")
            return False

        if response.status_code >= 400:
            logger.debug(f"HTTP error {response.status_code} for {url}")
            return False

        return True

    except requests.RequestException as e:
        logger.debug(f"Could not check accessibility for {url}: {str(e)}")
        # Don't exclude on check failure - might still be accessible
        return True


def fetch_page_title(url: str, timeout: int = 5) -> Optional[str]:
    """
    Fetch page title from URL.

    Args:
        url: URL to fetch
        timeout: Request timeout

    Returns:
        Page title or None
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)

        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title')
            if title:
                return title.text.strip()

        return None

    except Exception as e:
        logger.debug(f"Could not fetch title for {url}: {str(e)}")
        return None


class GoogleSearchClient:
    """Google Search client for finding sources."""

    def __init__(self):
        """Initialize Google Search client."""
        logger.info("Google Search client initialized")

    def search(
        self,
        query: str,
        max_results: int = 100,
        filter_paywalls: bool = True,
        check_accessibility: bool = True
    ) -> List[SearchResult]:
        """
        Search Google for sources.

        Args:
            query: Search query
            max_results: Maximum number of results (up to 100)
            filter_paywalls: Whether to filter out paywalled sources
            check_accessibility: Whether to check if URLs are accessible

        Returns:
            List of SearchResult objects
        """
        logger.info(f"Searching Google: '{query}' (max_results={max_results})")

        try:
            results = []
            seen_urls = set()

            # Google search with rate limiting
            search_results = google_search(
                query,
                num_results=max_results,
                advanced=True,
                sleep_interval=2,  # Be nice to Google
                lang='en'
            )

            for idx, result in enumerate(search_results):
                if len(results) >= max_results:
                    break

                # Extract URL and title
                url = result.url if hasattr(result, 'url') else str(result)
                title = result.title if hasattr(result, 'title') else result.description if hasattr(result, 'description') else ""
                description = result.description if hasattr(result, 'description') else ""

                # Skip duplicates
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # Filter paywalls
                if filter_paywalls and is_paywall_domain(url):
                    logger.debug(f"Skipping paywalled source: {url}")
                    continue

                # Check accessibility
                if check_accessibility and not check_url_accessible(url):
                    logger.debug(f"Skipping inaccessible source: {url}")
                    continue

                # Detect source type
                source_type = detect_source_type(url)

                # Get title if not available
                if not title:
                    title = fetch_page_title(url) or f"Source {idx + 1}"

                # Create search result
                search_result = SearchResult(
                    title=title[:200],  # Limit title length
                    url=url,
                    content_snippet=description[:500] if description else "",
                    source_type=source_type,
                    score=1.0 - (idx / max_results)  # Higher rank = higher score
                )

                results.append(search_result)

                # Rate limiting
                if idx % 10 == 0 and idx > 0:
                    time.sleep(1)  # Extra pause every 10 results

            logger.info(f"Found {len(results)} results for query: '{query}'")

            return results

        except Exception as e:
            logger.error(f"Google search failed: {str(e)}")
            raise
