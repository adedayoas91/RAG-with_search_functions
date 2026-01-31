"""
Source filtering and relevance scoring.

This module provides functions to filter search results by relevance
and deduplicate sources.
"""

from typing import List, Set, Optional
import numpy as np
from openai import OpenAI

from src.ingestion.web_search import SearchResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def filter_sources_by_relevance(
    query: str,
    sources: List[SearchResult],
    openai_client: Optional[OpenAI] = None,
    threshold: float = 0.6
) -> List[SearchResult]:
    """
    Filter sources by semantic relevance to query.

    Uses OpenAI embeddings to compute similarity between
    the query and each source's content snippet.

    Args:
        query: User's search query
        sources: List of SearchResult objects
        openai_client: Optional OpenAI client (if None, uses keyword matching)
        threshold: Minimum similarity score (0.0-1.0) to keep a source

    Returns:
        Filtered list of SearchResult objects above threshold

    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI(api_key="...")
        >>> sources = [...]
        >>> filtered = filter_sources_by_relevance("AI safety", sources, client, 0.6)
        >>> print(f"Kept {len(filtered)} of {len(sources)} sources")
    """
    if not sources:
        logger.warning("No sources to filter")
        return []

    logger.info(f"Filtering {len(sources)} sources with threshold {threshold}")

    # If no OpenAI client provided, use simple keyword matching
    if openai_client is None:
        logger.warning("No OpenAI client provided, using keyword-based filtering")
        return _filter_by_keywords(query, sources, threshold)

    try:
        # Generate embeddings for query and sources
        query_lower = query.lower()
        source_texts = [s.content_snippet for s in sources]

        # Create embedding request
        all_texts = [query] + source_texts

        logger.debug(f"Generating embeddings for query and {len(source_texts)} sources")
        response = openai_client.embeddings.create(
            input=all_texts,
            model="text-embedding-3-small"
        )

        # Extract embeddings
        embeddings = [np.array(item.embedding) for item in response.data]
        query_embedding = embeddings[0]
        source_embeddings = embeddings[1:]

        # Calculate cosine similarities
        similarities = []
        for source_emb in source_embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, source_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(source_emb)
            )
            similarities.append(float(similarity))

        # Filter sources above threshold
        filtered_sources = []
        for source, similarity in zip(sources, similarities):
            if similarity >= threshold:
                # Update source score to be the similarity score
                source.score = similarity
                filtered_sources.append(source)
                logger.debug(
                    f"Kept: {source.title[:50]}... (similarity: {similarity:.3f})"
                )
            else:
                logger.debug(
                    f"Filtered out: {source.title[:50]}... (similarity: {similarity:.3f})"
                )

        # Sort by similarity (descending)
        filtered_sources.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            f"Filtered to {len(filtered_sources)} sources "
            f"(removed {len(sources) - len(filtered_sources)})"
        )

        return filtered_sources

    except Exception as e:
        logger.error(f"Error filtering sources: {str(e)}")
        # Fallback: return all sources if filtering fails
        return sources


def _filter_by_keywords(
    query: str,
    sources: List[SearchResult],
    threshold: float = 0.5
) -> List[SearchResult]:
    """
    Simple keyword-based filtering fallback.

    Counts how many query words appear in the source snippet.
    """
    query_words = set(query.lower().split())

    filtered_sources = []
    for source in sources:
        snippet_words = set(source.content_snippet.lower().split())
        # Calculate overlap score
        overlap = len(query_words & snippet_words) / max(len(query_words), 1)

        if overlap >= threshold:
            source.score = overlap
            filtered_sources.append(source)

    # Sort by score
    filtered_sources.sort(key=lambda x: x.score, reverse=True)

    logger.info(f"Keyword filtering kept {len(filtered_sources)} of {len(sources)} sources")
    return filtered_sources


def deduplicate_sources(sources: List[SearchResult]) -> List[SearchResult]:
    """
    Remove duplicate sources based on URL.

    Args:
        sources: List of SearchResult objects

    Returns:
        Deduplicated list of SearchResult objects
    """
    if not sources:
        return []

    seen_urls: Set[str] = set()
    unique_sources = []

    for source in sources:
        # Normalize URL (remove trailing slashes, fragments, etc.)
        normalized_url = source.url.rstrip('/').split('#')[0].split('?')[0]

        if normalized_url not in seen_urls:
            seen_urls.add(normalized_url)
            unique_sources.append(source)
        else:
            logger.debug(f"Duplicate URL filtered: {source.url}")

    removed = len(sources) - len(unique_sources)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate sources")

    return unique_sources


def rank_sources_by_type(
    sources: List[SearchResult],
    preferred_types: List[str] = None
) -> List[SearchResult]:
    """
    Rank sources with preference for certain types.

    Args:
        sources: List of SearchResult objects
        preferred_types: List of preferred types in order (e.g., ['pdf', 'article', 'video'])

    Returns:
        Reordered list of SearchResult objects
    """
    if not preferred_types:
        preferred_types = ['pdf', 'article', 'video']

    # Create type priority map
    type_priority = {t: i for i, t in enumerate(preferred_types)}

    # Sort by type priority (lower is better) then by score (higher is better)
    sorted_sources = sorted(
        sources,
        key=lambda s: (type_priority.get(s.source_type, 999), -s.score)
    )

    return sorted_sources
