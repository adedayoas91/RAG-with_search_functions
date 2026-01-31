"""
Generation module for Agentic RAG System.

This package provides functions for query expansion and RAG-based answer generation.
"""

from .generate import generate_multi_query, DEFAULT_MULTI_QUERY_PROMPT
from .answer_generator import RAGAnswerGenerator, GeneratedAnswer, RAG_SYSTEM_PROMPT

__all__ = [
    "generate_multi_query",
    "DEFAULT_MULTI_QUERY_PROMPT",
    "RAGAnswerGenerator",
    "GeneratedAnswer",
    "RAG_SYSTEM_PROMPT",
]
