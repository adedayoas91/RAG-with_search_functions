"""
Source summarization for user approval.

This module provides async functions to generate quick summaries of sources
using LLMs.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional
from openai import OpenAI, AsyncOpenAI

from src.ingestion.web_search import SearchResult
from src.utils.logging_config import get_logger
from src.utils.cost_tracker import CostTracker

logger = get_logger(__name__)


@dataclass
class SourceSummary:
    """Summary of a search result source."""

    source: SearchResult
    summary: str
    relevance_score: float
    estimated_tokens: int = 0


async def summarize_source(
    source: SearchResult,
    query: str,
    client: AsyncOpenAI,
    cost_tracker: Optional[CostTracker] = None,
    model: str = "gpt-4o-mini"
) -> SourceSummary:
    """
    Generate a quick summary of a source for user approval.

    Args:
        source: SearchResult to summarize
        query: Original user query for context
        client: Async OpenAI client
        cost_tracker: Optional cost tracker
        model: Model to use for summarization

    Returns:
        SourceSummary object with summary and metadata

    Example:
        >>> client = AsyncOpenAI(api_key="...")
        >>> source = SearchResult(...)
        >>> summary = await summarize_source(source, "AI safety", client)
        >>> print(summary.summary)
    """
    try:
        # Create prompt
        prompt = f"""Summarize this content in 2-3 concise sentences, focusing on its relevance to the query: "{query}"

Title: {source.title}
Content: {source.content_snippet}

Provide a brief, informative summary."""

        # Call OpenAI API
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise summarization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        summary_text = response.choices[0].message.content.strip()

        # Track cost if tracker provided
        if cost_tracker:
            cost_tracker.track_openai_call(
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                operation="summarization",
                metadata={"source_url": source.url}
            )

        logger.debug(f"Summarized: {source.title[:50]}...")

        return SourceSummary(
            source=source,
            summary=summary_text,
            relevance_score=source.score,
            estimated_tokens=response.usage.total_tokens
        )

    except Exception as e:
        logger.error(f"Error summarizing source {source.url}: {str(e)}")
        # Fallback to content snippet
        return SourceSummary(
            source=source,
            summary=source.content_snippet[:200] + "...",
            relevance_score=source.score,
            estimated_tokens=0
        )


async def summarize_sources_batch(
    sources: List[SearchResult],
    query: str,
    client: AsyncOpenAI,
    cost_tracker: Optional[CostTracker] = None,
    model: str = "gpt-4o-mini"
) -> List[SourceSummary]:
    """
    Summarize multiple sources concurrently.

    Args:
        sources: List of SearchResult objects
        query: Original user query
        client: Async OpenAI client
        cost_tracker: Optional cost tracker
        model: Model to use for summarization

    Returns:
        List of SourceSummary objects

    Example:
        >>> client = AsyncOpenAI(api_key="...")
        >>> sources = [...]
        >>> summaries = await summarize_sources_batch(sources, "AI safety", client)
    """
    logger.info(f"Summarizing {len(sources)} sources...")

    # Create tasks for concurrent execution
    tasks = [
        summarize_source(source, query, client, cost_tracker, model)
        for source in sources
    ]

    # Execute all tasks concurrently
    summaries = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out any errors
    valid_summaries = []
    for i, result in enumerate(summaries):
        if isinstance(result, Exception):
            logger.error(f"Failed to summarize source {i}: {str(result)}")
            # Create fallback summary
            valid_summaries.append(
                SourceSummary(
                    source=sources[i],
                    summary=sources[i].content_snippet[:200] + "...",
                    relevance_score=sources[i].score,
                    estimated_tokens=0
                )
            )
        else:
            valid_summaries.append(result)

    logger.info(f"Successfully summarized {len(valid_summaries)} sources")

    return valid_summaries


def summarize_sources_sync(
    sources: List[SearchResult],
    query: str,
    client: OpenAI,
    cost_tracker: Optional[CostTracker] = None,
    model: str = "gpt-4o-mini"
) -> List[SourceSummary]:
    """
    Synchronous wrapper for batch summarization.

    Args:
        sources: List of SearchResult objects
        query: Original user query
        client: Sync OpenAI client (will be converted to async)
        cost_tracker: Optional cost tracker
        model: Model to use for summarization

    Returns:
        List of SourceSummary objects
    """
    # Create async client from sync client
    async_client = AsyncOpenAI(api_key=client.api_key)

    # Run async function in event loop
    return asyncio.run(
        summarize_sources_batch(sources, query, async_client, cost_tracker, model)
    )
