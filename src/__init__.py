"""
Agentic RAG System - Source Package.

This is the main package for the Agentic RAG system, providing modules for
ingestion, vector storage, and generation.
"""

__version__ = "1.0.0"

# Import key components for easy access
from . import ingestion
from . import vectorstore
from . import generation
from . import utils

__all__ = [
    "ingestion",
    "vectorstore",
    "generation",
    "utils",
]
