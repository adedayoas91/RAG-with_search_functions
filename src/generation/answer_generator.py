"""
RAG answer generation using retrieved context.

This module provides functions to generate answers based on retrieved context.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from openai import OpenAI
from langchain_core.documents import Document

from src.utils.logging_config import get_logger
from src.utils.cost_tracker import CostTracker

logger = get_logger(__name__)


# RAG system prompt with numeric citations
RAG_SYSTEM_PROMPT = """You are a knowledgeable research assistant. Answer the user's question using ONLY the provided context from retrieved sources.

Guidelines:
1. Base your answer strictly on the provided context
2. Cite sources using NUMERIC references like [1], [2], [1,3], or [2-5]
3. Use [1] for a single source, [1,3] for multiple non-consecutive sources, and [2-5] for consecutive ranges
4. If the context doesn't contain enough information, say so
5. Be concise but comprehensive
6. Use a professional, informative tone
7. Do not make up information not present in the context
8. DO NOT include a "Sources:" or "References:" section at the end - this will be added automatically

Context with source numbers:
{context}"""


@dataclass
class GeneratedAnswer:
    """Generated answer with metadata."""

    answer: str
    sources: List[str]
    tokens_used: int
    cost: float
    model: str


class RAGAnswerGenerator:
    """Generate answers using RAG (Retrieval-Augmented Generation)."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        cost_tracker: Optional[CostTracker] = None
    ):
        """
        Initialize RAG answer generator.

        Args:
            client: OpenAI client instance
            model: Model to use for generation
            cost_tracker: Optional cost tracker

        Example:
            >>> from openai import OpenAI
            >>> client = OpenAI(api_key="...")
            >>> generator = RAGAnswerGenerator(client)
        """
        self.client = client
        self.model = model
        self.cost_tracker = cost_tracker

        logger.info(f"RAG answer generator initialized with model: {model}")

    def generate_answer(
        self,
        query: str,
        context_documents: List[Document],
        temperature: float = 0.2,
        max_tokens: int = 2000
    ) -> GeneratedAnswer:
        """
        Generate answer using retrieved context.

        Args:
            query: User's question
            context_documents: List of retrieved documents
            temperature: Generation temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            GeneratedAnswer object with answer and metadata

        Example:
            >>> docs = [Document(page_content="...", metadata={"source": "url"})]
            >>> answer = generator.generate_answer("What is AI safety?", docs)
            >>> print(answer.answer)
        """
        logger.info(f"Generating answer for query: '{query}'")

        try:
            # Format context
            context, sources = self._format_context(context_documents)

            # Create system prompt with context
            system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

            # Generate answer
            logger.debug(f"Calling {self.model} with {len(context)} chars of context")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            answer_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens

            # Append sources list at the end
            answer_with_sources = self._append_sources_list(answer_text, sources)

            # Track cost
            cost = 0.0
            if self.cost_tracker:
                cost = self.cost_tracker.track_openai_call(
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    operation="answer_generation",
                    metadata={"query": query}
                )

            logger.info(
                f"Generated answer: {len(answer_with_sources)} chars, "
                f"{tokens_used} tokens, ${cost:.4f}"
            )

            return GeneratedAnswer(
                answer=answer_with_sources,
                sources=sources,
                tokens_used=tokens_used,
                cost=cost,
                model=self.model
            )

        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            raise

    def _format_context(
        self,
        documents: List[Document],
        max_context_length: int = 8000
    ) -> tuple[str, List[str]]:
        """
        Format documents into context string with numeric citations.

        Args:
            documents: List of Document objects
            max_context_length: Maximum context length in characters

        Returns:
            Tuple of (formatted_context, list_of_source_urls)
        """
        if not documents:
            return "No context available.", []

        context_parts = []
        sources = []
        current_length = 0

        for i, doc in enumerate(documents, 1):
            source_url = doc.metadata.get("source", "Unknown")
            content = doc.page_content

            # Format with numeric source reference
            formatted = f"\n[{i}] {content}\n"

            # Check if adding this would exceed max length
            if current_length + len(formatted) > max_context_length:
                logger.warning(
                    f"Context truncated at {current_length} chars "
                    f"(max: {max_context_length})"
                )
                break

            context_parts.append(formatted)
            sources.append(source_url)
            current_length += len(formatted)

        context = "\n---\n".join(context_parts)

        logger.debug(f"Formatted context: {current_length} chars from {len(sources)} sources")

        return context, sources

    def _append_sources_list(self, answer: str, sources: List[str]) -> str:
        """
        Append numbered sources list to the end of the answer.

        Args:
            answer: Generated answer text
            sources: List of source URLs

        Returns:
            Answer with sources list appended
        """
        if not sources:
            return answer

        # Build sources section
        sources_section = "\n\n## Sources\n"

        for i, source in enumerate(sources, 1):
            sources_section += f"[{i}] {source}\n"

        return answer + sources_section
