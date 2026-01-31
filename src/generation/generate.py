"""
Multi-query generation for enhanced retrieval.

This module provides functions to expand user queries into multiple related queries
for better RAG retrieval coverage.
"""

from typing import List, Optional
from openai import OpenAI

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Default prompt template for multi-query generation
DEFAULT_MULTI_QUERY_PROMPT = """You are a knowledgeable research assistant.
For the given question, propose up to 5 related questions to assist in finding comprehensive information.
Provide concise, single-topic questions that cover various aspects of the topic.
Ensure each question is complete and directly related to the original inquiry.
List each question on a separate line without numbering."""


def generate_multi_query(
    query: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    prompt_template: Optional[str] = None,
    max_queries: int = 5
) -> List[str]:
    """
    Generate multiple related queries from a single user query.

    This uses an LLM to expand a query into multiple related queries,
    which helps improve retrieval coverage in RAG systems.

    Args:
        query: Original user query
        client: OpenAI client instance
        model: Model to use for generation (default: gpt-4o-mini)
        prompt_template: Custom prompt template (default: DEFAULT_MULTI_QUERY_PROMPT)
        max_queries: Maximum number of queries to generate

    Returns:
        List of related query strings (includes original query)

    Example:
        >>> client = OpenAI(api_key="...")
        >>> queries = generate_multi_query("What is AI safety?", client)
        >>> print(queries)
        ['What is AI safety?', 'How do we ensure AI systems are safe?', ...]
    """
    try:
        # Use default prompt if none provided
        system_prompt = prompt_template or DEFAULT_MULTI_QUERY_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        logger.debug(f"Generating multi-query expansion for: {query}")

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )

        # Extract content and split into lines
        content = response.choices[0].message.content
        queries = [q.strip() for q in content.split("\n") if q.strip()]

        # Filter out empty queries and limit to max_queries
        queries = [q for q in queries if len(q) > 10][:max_queries]

        # Always include the original query
        if query not in queries:
            queries.insert(0, query)

        logger.info(f"Generated {len(queries)} queries (including original)")
        logger.debug(f"Queries: {queries}")

        return queries

    except Exception as e:
        logger.error(f"Error generating multi-query: {str(e)}")
        # Fallback to just the original query
        return [query]