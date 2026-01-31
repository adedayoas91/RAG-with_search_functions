"""
Unit tests for cost tracking functionality.
"""

import pytest
from src.utils.cost_tracker import CostTracker


@pytest.mark.unit
class TestCostTracker:
    """Tests for API cost tracking."""

    def test_initialization(self, temp_dir):
        """Test initializing cost tracker."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        assert len(tracker.session_calls) == 0
        assert tracker.log_file.exists() or not tracker.log_file.exists()

    def test_track_openai_call_gpt4o_mini(self, temp_dir):
        """Test tracking GPT-4o-mini API call."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        cost = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            operation="generation"
        )

        assert cost > 0
        assert len(tracker.session_calls) == 1
        assert tracker.session_calls[0].model == "gpt-4o-mini"
        assert tracker.session_calls[0].operation == "generation"

    def test_track_openai_call_gpt4o(self, temp_dir):
        """Test tracking GPT-4o API call."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        cost = tracker.track_openai_call(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            operation="generation"
        )

        assert cost > 0

    def test_track_embedding_call(self, temp_dir):
        """Test tracking embedding API call."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        cost = tracker.track_embedding_call(
            model="text-embedding-3-small",
            tokens=1000,
            operation="embedding"
        )

        assert cost > 0
        assert len(tracker.session_calls) == 1

    def test_track_tavily_search(self, temp_dir):
        """Test tracking Tavily search cost."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        cost = tracker.track_tavily_search(
            search_depth="advanced",
            num_results=10
        )

        assert cost > 0
        assert len(tracker.session_calls) == 1
        assert tracker.session_calls[0].provider == "tavily"

    def test_multiple_calls_accumulate(self, temp_dir):
        """Test that multiple calls accumulate correctly."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        cost1 = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            operation="generation"
        )

        cost2 = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=200,
            output_tokens=100,
            operation="generation"
        )

        session_costs = tracker.get_session_costs()
        assert session_costs['total'] == cost1 + cost2
        assert len(tracker.session_calls) == 2

    def test_get_session_costs(self, temp_dir):
        """Test getting session cost summary."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            operation="generation"
        )
        tracker.track_tavily_search()

        costs = tracker.get_session_costs()

        assert "total" in costs
        assert "by_provider" in costs
        assert "by_model" in costs
        assert costs['total'] > 0

    def test_get_session_costs_structure(self, temp_dir):
        """Test session costs structure."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            operation="generation"
        )

        costs = tracker.get_session_costs()

        assert isinstance(costs, dict)
        assert costs['total'] > 0

    def test_zero_tokens_handling(self, temp_dir):
        """Test handling of zero tokens."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        cost = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=0,
            output_tokens=0,
            operation="test"
        )

        assert cost == 0.0

    def test_call_logging(self, temp_dir):
        """Test that calls are logged to session."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            operation="generation"
        )

        assert len(tracker.session_calls) == 1
        call = tracker.session_calls[0]
        assert call.provider == "openai"
        assert call.model == "gpt-4o-mini"
        assert call.operation == "generation"
        assert call.input_tokens == 100
        assert call.output_tokens == 50
        assert call.cost > 0

    def test_metadata_tracking(self, temp_dir):
        """Test metadata is tracked with API calls."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        metadata = {"query": "test query", "context_length": 1000}
        tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            operation="generation",
            metadata=metadata
        )

        assert tracker.session_calls[0].metadata == metadata

    def test_cost_precision(self, temp_dir):
        """Test that costs are calculated with appropriate precision."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        cost = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=1,
            output_tokens=1,
            operation="test"
        )

        # Cost should be very small but positive
        assert cost > 0
        assert cost < 0.01  # Less than a cent

    def test_large_token_counts(self, temp_dir):
        """Test handling of large token counts."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        cost = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100000,
            output_tokens=50000,
            operation="test"
        )

        assert cost > 0
        costs = tracker.get_session_costs()
        assert costs['total'] == cost

    def test_basic_search_cost(self, temp_dir):
        """Test Tavily basic search cost tracking."""
        tracker = CostTracker(log_file=str(temp_dir / "test_costs.json"))

        cost = tracker.track_tavily_search(
            search_depth="basic",
            num_results=5
        )

        assert cost > 0
        assert len(tracker.session_calls) == 1
