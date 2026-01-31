"""
Web article loading and extraction.

This module provides functions to fetch and extract clean text from web articles.
"""

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from typing import Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def fetch_html(url: str, timeout: int = 15) -> str:
    """
    Fetch HTML content from URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        HTML content as string

    Raises:
        requests.RequestException: If fetch fails
    """
    logger.debug(f"Fetching HTML from: {url}")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding  # Handle encoding properly

        return response.text

    except Exception as e:
        logger.error(f"Failed to fetch HTML from {url}: {str(e)}")
        raise


def clean_html_content(html: str) -> str:
    """
    Extract and clean main content from HTML.

    Args:
        html: Raw HTML string

    Returns:
        Cleaned text content
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'iframe', 'noscript']):
            element.decompose()

        # Try to find main content in common article containers
        content = None
        content_selectors = [
            'article',
            '[role="main"]',
            'main',
            '.article-content',
            '.post-content',
            '.entry-content',
            '#content',
            '.content'
        ]

        for selector in content_selectors:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 200:
                break

        # If no main content found, use body
        if not content:
            content = soup.body if soup.body else soup

        # Extract text
        text = content.get_text(separator='\n', strip=True)

        # Clean up text
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 3:  # Filter out very short lines
                lines.append(line)

        text = '\n'.join(lines)

        # Remove excessive newlines
        import re
        text = re.sub(r'\n\n+', '\n\n', text)

        return text.strip()

    except Exception as e:
        logger.error(f"Failed to clean HTML content: {str(e)}")
        raise


def extract_title(html: str) -> Optional[str]:
    """
    Extract title from HTML.

    Args:
        html: Raw HTML string

    Returns:
        Title string or None
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Try various title sources
        if soup.title and soup.title.string:
            return soup.title.string.strip()

        # Try meta og:title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()

        # Try h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)

        return None

    except:
        return None


def load_article(url: str) -> Document:
    """
    Load web article as a LangChain Document.

    This is the main function to use for loading articles into the RAG system.

    Args:
        url: URL of web article

    Returns:
        LangChain Document with extracted text and metadata

    Raises:
        Exception: If fetch or extraction fails

    Example:
        >>> doc = load_article("https://example.com/article")
        >>> print(f"Loaded {len(doc.page_content)} characters")
        >>> print(doc.metadata['title'])
    """
    logger.info(f"Loading article: {url}")

    try:
        # Fetch HTML
        html = fetch_html(url)

        # Extract title
        title = extract_title(html)

        # Extract and clean content
        text = clean_html_content(html)

        if not text:
            raise ValueError(f"No text extracted from article: {url}")

        if len(text) < 100:
            logger.warning(f"Very little text extracted ({len(text)} chars)")

        # Create Document with metadata
        doc = Document(
            page_content=text,
            metadata={
                "source": url,
                "source_type": "article",
                "title": title,
                "content_length": len(text)
            }
        )

        logger.info(f"Successfully loaded article: {len(text)} characters from {url}")

        return doc

    except Exception as e:
        logger.error(f"Failed to load article {url}: {str(e)}")
        raise
