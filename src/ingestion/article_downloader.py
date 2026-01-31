"""
Journal article downloader for RAG system.

This module provides functionality to download journal articles and PDFs
from URLs using wget, organizing them by query topic.
"""

import os
import subprocess
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup

from src.ingestion.web_search import SearchResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Sanitize filename to remove invalid characters.

    Args:
        filename: Original filename
        max_length: Maximum length for filename

    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')

    # Limit length
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext

    return filename


def create_query_directory(base_dir: str, query: str) -> Path:
    """
    Create a directory for downloaded articles based on query.

    Args:
        base_dir: Base directory (e.g., './data')
        query: User's research query

    Returns:
        Path to created directory
    """
    # Sanitize query to create directory name
    dir_name = sanitize_filename(query.lower().replace(' ', '_'))[:50]

    # Create full path
    query_dir = Path(base_dir) / "downloads" / dir_name
    query_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created download directory: {query_dir}")

    return query_dir


def is_downloadable_article(url: str) -> bool:
    """
    Check if URL is likely a downloadable article/PDF.

    Args:
        url: URL to check

    Returns:
        True if URL appears to be a downloadable document
    """
    url_lower = url.lower()

    # Check for PDF extension
    if url_lower.endswith('.pdf'):
        return True

    # Check for common academic/journal domains
    academic_domains = [
        'arxiv.org',
        'doi.org',
        'ncbi.nlm.nih.gov',
        'springer.com',
        'sciencedirect.com',
        'wiley.com',
        'nature.com',
        'science.org',
        'ieee.org',
        'acm.org',
        'jstor.org',
        'researchgate.net',
        'biorxiv.org',
        'medrxiv.org',
        'ssrn.com',
        'papers.ssrn.com'
    ]

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    return any(academic_domain in domain for academic_domain in academic_domains)


def extract_filename_from_url(url: str, source_title: Optional[str] = None) -> str:
    """
    Extract a reasonable filename from URL or source title.

    Args:
        url: URL of the article
        source_title: Optional title of the article

    Returns:
        Suggested filename
    """
    # Try to get filename from URL
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # If URL ends with .pdf, use that
    if path.endswith('.pdf'):
        filename = os.path.basename(path)
        return sanitize_filename(filename)

    # Otherwise, use source title
    if source_title:
        filename = sanitize_filename(source_title) + '.pdf'
        return filename

    # Fallback: use domain and path
    domain = parsed.netloc.replace('www.', '')
    filename = f"{domain}_{os.path.basename(path)}.pdf"
    return sanitize_filename(filename)


def parse_and_save_article(
    url: str,
    output_dir: Path,
    title: str,
    timeout: int = 30
) -> Tuple[bool, Optional[Path]]:
    """
    Parse HTML article and save as text file.

    Args:
        url: URL to parse
        output_dir: Directory to save file
        title: Article title for filename
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, file_path)
    """
    try:
        # Fetch HTML
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        logger.info(f"Parsing article: {url}")
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()

        # Extract text from main content areas
        main_content = None

        # Try to find main content area
        for selector in ['article', 'main', '[role="main"]', '.article-content', '.post-content', '.entry-content']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.body if soup.body else soup

        # Extract text
        text = main_content.get_text(separator='\n', strip=True)

        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n\n'.join(lines)

        # Check if we got meaningful content
        if len(cleaned_text) < 200:
            logger.warning(f"Very little text extracted ({len(cleaned_text)} chars) from {url}")
            return False, None

        # Create filename from title
        safe_title = sanitize_filename(title)
        if not safe_title.endswith('.txt'):
            safe_title += '.txt'

        output_path = output_dir / safe_title

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {url}\n")
            f.write(f"Title: {title}\n")
            f.write("=" * 80 + "\n\n")
            f.write(cleaned_text)

        file_size = output_path.stat().st_size
        logger.info(f"Successfully parsed and saved: {output_path} ({file_size / 1024:.1f} KB)")

        return True, output_path

    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {str(e)}")
        return False, None
    except Exception as e:
        logger.error(f"Error parsing {url}: {str(e)}")
        return False, None


def download_article_wget(
    url: str,
    output_dir: Path,
    filename: Optional[str] = None,
    timeout: int = 60
) -> Tuple[bool, Optional[Path]]:
    """
    Download article using wget.

    Args:
        url: URL to download
        output_dir: Directory to save file
        filename: Optional custom filename
        timeout: Download timeout in seconds

    Returns:
        Tuple of (success, file_path)
    """
    try:
        # Check if wget is available
        result = subprocess.run(
            ['which', 'wget'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error("wget is not installed. Please install wget first.")
            return False, None

        # Prepare wget command
        output_path = output_dir / (filename or 'downloaded_article.pdf')

        wget_cmd = [
            'wget',
            '--timeout', str(timeout),
            '--tries', '3',
            '--user-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            '--no-check-certificate',  # Some academic sites have cert issues
            '-O', str(output_path),
            url
        ]

        logger.info(f"Downloading: {url}")
        logger.debug(f"Command: {' '.join(wget_cmd)}")

        result = subprocess.run(
            wget_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10
        )

        if result.returncode == 0 and output_path.exists():
            file_size = output_path.stat().st_size

            # Check if file is too small (likely error page)
            if file_size < 1024:  # Less than 1KB
                logger.warning(f"Downloaded file is suspiciously small ({file_size} bytes), likely an error")
                output_path.unlink()
                return False, None

            logger.info(f"Successfully downloaded: {output_path} ({file_size / 1024:.1f} KB)")
            return True, output_path
        else:
            logger.error(f"wget failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            return False, None

    except subprocess.TimeoutExpired:
        logger.error(f"Download timed out after {timeout} seconds")
        return False, None
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False, None


def _download_single_article(source: SearchResult, query_dir: Path) -> Optional[Tuple[SearchResult, Path]]:
    """
    Download or parse a single article (helper for parallel processing).

    Args:
        source: SearchResult object
        query_dir: Directory to save downloads

    Returns:
        Tuple of (source, file_path) if successful, None otherwise
    """
    # Check if source is downloadable (PDF)
    if is_downloadable_article(source.url):
        # Extract filename
        filename = extract_filename_from_url(source.url, source.title)

        # Download PDF with wget
        success, file_path = download_article_wget(
            url=source.url,
            output_dir=query_dir,
            filename=filename
        )

        if success and file_path:
            return (source, file_path)
        else:
            # If download fails, try parsing as regular article
            logger.info(f"Download failed, trying to parse as article: {source.url}")

    # Parse as regular HTML article and save as text
    success, file_path = parse_and_save_article(
        url=source.url,
        output_dir=query_dir,
        title=source.title
    )

    if success and file_path:
        return (source, file_path)

    return None


def download_articles_from_sources(
    sources: List[SearchResult],
    query: str,
    base_dir: str = "./data",
    min_downloads: int = 10,
    max_workers: int = 5
) -> List[Tuple[SearchResult, Path]]:
    """
    Download or parse articles from search results in parallel, ensuring minimum successful saves.

    For downloadable PDFs: downloads with wget
    For regular articles: parses HTML and saves as .txt file

    Args:
        sources: List of SearchResult objects
        query: User's research query
        base_dir: Base directory for downloads
        min_downloads: Minimum number of successful saves required
        max_workers: Maximum number of parallel worker threads

    Returns:
        List of tuples (source, saved_file_path) for successful saves
    """
    # Create query-specific directory
    query_dir = create_query_directory(base_dir, query)

    saved_articles = []
    attempted = 0

    logger.info(f"Target: {min_downloads} successful saves (using {max_workers} parallel workers)")
    logger.info(f"Processing {len(sources)} sources (PDFs will be downloaded, articles will be parsed)")

    # Process in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for all sources (not just downloadable ones)
        future_to_source = {}
        sources_to_process = sources[:min_downloads * 3]  # Try up to 3x the target

        for source in sources_to_process:
            if len(saved_articles) >= min_downloads:
                break

            future = executor.submit(_download_single_article, source, query_dir)
            future_to_source[future] = source
            attempted += 1

        # Collect results as they complete
        for future in as_completed(future_to_source):
            source = future_to_source[future]

            try:
                result = future.result()
                if result:
                    saved_articles.append(result)
                    file_type = "PDF" if result[1].suffix == '.pdf' else "TXT"
                    logger.info(f"✓ Saved {file_type} ({len(saved_articles)}/{min_downloads}): {source.title[:60]}...")

                    # Submit more tasks if we haven't reached the target
                    if len(saved_articles) < min_downloads and attempted < len(sources_to_process):
                        next_source = sources_to_process[attempted]
                        future = executor.submit(_download_single_article, next_source, query_dir)
                        future_to_source[future] = next_source
                        attempted += 1
                else:
                    logger.debug(f"✗ Failed: {source.url}")

            except Exception as e:
                logger.warning(f"Error processing {source.url}: {str(e)}")

            # Early exit if we've reached the target
            if len(saved_articles) >= min_downloads:
                logger.info(f"Reached target of {min_downloads} saves, stopping")
                break

    # Count PDFs vs TXT files
    pdf_count = sum(1 for _, path in saved_articles if path.suffix == '.pdf')
    txt_count = sum(1 for _, path in saved_articles if path.suffix == '.txt')

    if len(saved_articles) < min_downloads:
        logger.warning(
            f"Only saved {len(saved_articles)} articles (target: {min_downloads}). "
            f"Tried {attempted} sources out of {len(sources)} available. "
            f"PDFs: {pdf_count}, Articles: {txt_count}"
        )
    else:
        logger.info(
            f"Successfully saved {len(saved_articles)} articles (target: {min_downloads}). "
            f"PDFs: {pdf_count}, Articles: {txt_count}"
        )

    return saved_articles
