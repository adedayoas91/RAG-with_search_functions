"""
API cost tracking for Agentic RAG System.

This module provides real-time cost tracking for OpenAI, Tavily, and other API calls.
Costs are logged to a JSON file for historical analysis and dashboard display.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from threading import Lock

from config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class APICall:
    """Record of a single API call."""

    timestamp: str
    provider: str  # 'openai', 'tavily', etc.
    operation: str  # 'summarization', 'generation', 'embedding', 'search'
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class CostTracker:
    """Track API costs and maintain cost logs."""

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize cost tracker.

        Args:
            log_file: Path to cost log file (default: from config)
        """
        self.config = get_config()
        self.log_file = Path(
            log_file or self.config.paths.cost_log_file
        )

        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Session tracking
        self.session_calls: List[APICall] = []
        self.session_start_time = datetime.now()

        # Thread-safe logging
        self._lock = Lock()

        logger.info(f"CostTracker initialized with log file: {self.log_file}")

    def track_openai_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Track an OpenAI API call and calculate cost.

        Args:
            model: Model name (e.g., 'gpt-4o-mini')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation: Operation type (e.g., 'summarization', 'generation')
            metadata: Additional metadata to log

        Returns:
            Cost in USD
        """
        # Calculate cost
        input_cost_per_1k = self.config.get_model_cost(model, "input")
        output_cost_per_1k = self.config.get_model_cost(model, "output")

        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost

        # Create API call record
        api_call = APICall(
            timestamp=datetime.now().isoformat(),
            provider="openai",
            operation=operation,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=total_cost,
            metadata=metadata
        )

        # Log the call
        self._log_call(api_call)

        logger.debug(
            f"OpenAI call tracked: {operation} | "
            f"{model} | in:{input_tokens} out:{output_tokens} | "
            f"${total_cost:.4f}"
        )

        return total_cost

    def track_embedding_call(
        self,
        model: str,
        tokens: int,
        operation: str = "embedding",
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Track an embedding API call.

        Args:
            model: Embedding model name
            tokens: Number of tokens embedded
            operation: Operation type
            metadata: Additional metadata

        Returns:
            Cost in USD
        """
        # Get cost per 1k tokens
        cost_per_1k = self.config.get_model_cost(model, "input")
        total_cost = (tokens / 1000) * cost_per_1k

        # Create API call record
        api_call = APICall(
            timestamp=datetime.now().isoformat(),
            provider="openai" if "text-embedding" in model else "local",
            operation=operation,
            model=model,
            input_tokens=tokens,
            output_tokens=0,
            cost=total_cost,
            metadata=metadata
        )

        # Log the call
        self._log_call(api_call)

        if total_cost > 0:
            logger.debug(
                f"Embedding call tracked: {model} | "
                f"{tokens} tokens | ${total_cost:.4f}"
            )

        return total_cost

    def track_tavily_search(
        self,
        search_depth: str = "advanced",
        num_results: int = 10,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Track a Tavily search API call.

        Args:
            search_depth: 'basic' or 'advanced'
            num_results: Number of results requested
            metadata: Additional metadata

        Returns:
            Cost in USD
        """
        # Determine cost based on search depth
        if search_depth == "advanced":
            cost = self.config.cost.tavily_advanced_search
        else:
            cost = self.config.cost.tavily_basic_search

        # Create API call record
        api_call = APICall(
            timestamp=datetime.now().isoformat(),
            provider="tavily",
            operation="search",
            model=f"tavily-{search_depth}",
            input_tokens=0,
            output_tokens=0,
            cost=cost,
            metadata={
                **(metadata or {}),
                "search_depth": search_depth,
                "num_results": num_results
            }
        )

        # Log the call
        self._log_call(api_call)

        logger.debug(
            f"Tavily search tracked: {search_depth} | "
            f"{num_results} results | ${cost:.4f}"
        )

        return cost

    def _log_call(self, api_call: APICall) -> None:
        """
        Log API call to session and persistent storage.

        Args:
            api_call: API call record
        """
        with self._lock:
            # Add to session
            self.session_calls.append(api_call)

            # Append to log file
            try:
                # Load existing logs
                if self.log_file.exists():
                    with open(self.log_file, 'r') as f:
                        logs = json.load(f)
                else:
                    logs = []

                # Append new call
                logs.append(api_call.to_dict())

                # Save back to file
                with open(self.log_file, 'w') as f:
                    json.dump(logs, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to log API call: {str(e)}")

    def get_session_costs(self) -> Dict:
        """
        Get cost summary for current session.

        Returns:
            Dictionary with cost breakdown
        """
        total_cost = sum(call.cost for call in self.session_calls)
        by_provider = {}
        by_model = {}
        by_operation = {}

        for call in self.session_calls:
            # By provider
            by_provider[call.provider] = (
                by_provider.get(call.provider, 0.0) + call.cost
            )

            # By model
            by_model[call.model] = (
                by_model.get(call.model, 0.0) + call.cost
            )

            # By operation
            by_operation[call.operation] = (
                by_operation.get(call.operation, 0.0) + call.cost
            )

        return {
            "total": total_cost,
            "by_provider": by_provider,
            "by_model": by_model,
            "by_operation": by_operation,
            "num_calls": len(self.session_calls),
            "duration": (datetime.now() - self.session_start_time).total_seconds()
        }

    def get_total_costs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Get total costs from log file, optionally filtered by date range.

        Args:
            start_date: ISO format date string (YYYY-MM-DD)
            end_date: ISO format date string (YYYY-MM-DD)

        Returns:
            Dictionary with cost breakdown
        """
        if not self.log_file.exists():
            return {
                "total": 0.0,
                "by_provider": {},
                "by_model": {},
                "by_operation": {},
                "num_calls": 0
            }

        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)

            # Filter by date range if specified
            if start_date or end_date:
                filtered_logs = []
                for log in logs:
                    timestamp = log["timestamp"][:10]  # Get YYYY-MM-DD
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                    filtered_logs.append(log)
                logs = filtered_logs

            # Calculate totals
            total_cost = sum(log["cost"] for log in logs)
            by_provider = {}
            by_model = {}
            by_operation = {}

            for log in logs:
                provider = log["provider"]
                model = log["model"]
                operation = log["operation"]
                cost = log["cost"]

                by_provider[provider] = by_provider.get(provider, 0.0) + cost
                by_model[model] = by_model.get(model, 0.0) + cost
                by_operation[operation] = by_operation.get(operation, 0.0) + cost

            return {
                "total": total_cost,
                "by_provider": by_provider,
                "by_model": by_model,
                "by_operation": by_operation,
                "num_calls": len(logs)
            }

        except Exception as e:
            logger.error(f"Failed to load cost logs: {str(e)}")
            return {
                "total": 0.0,
                "by_provider": {},
                "by_model": {},
                "by_operation": {},
                "num_calls": 0
            }

    def reset_session(self) -> None:
        """Reset session tracking for a new session."""
        self.session_calls = []
        self.session_start_time = datetime.now()
        logger.debug("Cost tracker session reset")

    def get_session_summary(self) -> str:
        """
        Get formatted summary of session costs.

        Returns:
            Formatted string with cost breakdown
        """
        costs = self.get_session_costs()

        summary = f"\n=== Session Cost Summary ===\n"
        summary += f"Total Cost: ${costs['total']:.4f}\n"
        summary += f"Duration: {costs['duration']:.1f}s\n"
        summary += f"API Calls: {costs['num_calls']}\n\n"

        if costs['by_provider']:
            summary += "By Provider:\n"
            for provider, cost in sorted(
                costs['by_provider'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                summary += f"  {provider}: ${cost:.4f}\n"

        if costs['by_model']:
            summary += "\nBy Model:\n"
            for model, cost in sorted(
                costs['by_model'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                summary += f"  {model}: ${cost:.4f}\n"

        if costs['by_operation']:
            summary += "\nBy Operation:\n"
            for operation, cost in sorted(
                costs['by_operation'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                summary += f"  {operation}: ${cost:.4f}\n"

        summary += "="*30

        return summary
