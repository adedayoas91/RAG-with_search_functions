from typing import Dict, List, Any
from dataclasses import dataclass, asdict

from tools.search_tools import (
    search_wikipedia,
    search_tavily,
    search_serper,
    search_youtube,
    search_google,
)

from utils import logger


@dataclass
class SearchResult:
    """
    Normalized search result for downstream RAG pipelines
    """
    source: str
    query: str
    content: str


class SearchAgent:
    """
    Agent responsible for multi-source web retrieval.
    
    Design goals:
    - Uses ALL available search tools
    - Isolates tool failures
    - Produces normalized, RAG-ready outputs
    - Deterministic execution (important for production)
    """

    def __init__(self):
        self.tools = {
            "wikipedia": search_wikipedia,
            "tavily": search_tavily,
            "serper": search_serper,
            "youtube": search_youtube,
            "google": search_google,
        }

    def run(self, query: str) -> List[SearchResult]:
        """
        Execute the search agent over all tools.

        Args:
            query (str): User query

        Returns:
            List[SearchResult]: Normalized search results
        """
        results: List[SearchResult] = []

        for source, tool in self.tools.items():
            try:
                logger.info(f"[SearchAgent] Running {source} search")
                output = tool.run(query)

                results.append(
                    SearchResult(
                        source=source,
                        query=query,
                        content=output,
                    )
                )

            except Exception as e:
                # Hard isolation: one tool must not kill the agent
                logger.error(f"[SearchAgent] {source} failed: {e}")

        return results

    def run_as_dict(self, query: str) -> Dict[str, Any]:
        """
        Convenience method for APIs / JSON serialization
        """
        results = self.run(query)
        return {
            "query": query,
            "results": [asdict(r) for r in results],
        }
