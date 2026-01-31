"""
Unit tests for cost tracking functionality.
"""

import pytest
from unittest.mock import Mock
from src.utils.cost_tracker import CostTracker


@pytest.mark.unit
class TestCostTracker:
    """Tests for API cost tracking."""

    def test_initialization(self):
        """Test initializing cost tracker."""
        tracker = CostTracker()

        assert tracker.total_cost == 0.0
        assert len(tracker.call_history) == 0

    def test_track_openai_call_gpt4o_mini(self):
        """Test tracking GPT-4o-mini API call."""
        tracker = CostTracker()

        cost = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50
        )

        assert cost > 0
        assert tracker.total_cost == cost
        assert len(tracker.call_history) == 1
        assert tracker.call_history[0]["model"] == "gpt-4o-mini"

    def test_track_openai_call_gpt4o(self):
        """Test tracking GPT-4o API call."""
        tracker = CostTracker()

        cost = tracker.track_openai_call(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50
        )

        # GPT-4o should be more expensive than mini
        assert cost > 0
        assert tracker.total_cost == cost

    def test_track_openai_call_embedding(self):
        """Test tracking embedding API call."""
        tracker = CostTracker()

        cost = tracker.track_openai_call(
            model="text-embedding-3-small",
            input_tokens=1000,
            output_tokens=0
        )

        assert cost > 0
        assert tracker.total_cost == cost

    def test_track_tavily_search(self):
        """Test tracking Tavily search cost."""
        tracker = CostTracker()

        cost = tracker.track_tavily_search(num_searches=5)

        assert cost > 0
        assert tracker.total_cost == cost
        assert len(tracker.call_history) == 1
        assert tracker.call_history[0]["service"] == "tavily"

    def test_multiple_calls_accumulate(self):
        """Test that multiple calls accumulate correctly."""
        tracker = CostTracker()

        cost1 = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50
        )

        cost2 = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=200,
            output_tokens=100
        )

        assert tracker.total_cost == cost1 + cost2
        assert len(tracker.call_history) == 2

    def test_get_cost_breakdown(self):
        """Test getting cost breakdown by service."""
        tracker = CostTracker()

        tracker.track_openai_call(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        tracker.track_openai_call(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        tracker.track_tavily_search(num_searches=2)

        breakdown = tracker.get_cost_breakdown()

        assert "openai" in breakdown
        assert "tavily" in breakdown
        assert breakdown["openai"]["num_calls"] == 2
        assert breakdown["tavily"]["num_calls"] == 1

    def test_get_summary(self):
        """Test getting cost summary."""
        tracker = CostTracker()

        tracker.track_openai_call(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        tracker.track_tavily_search(num_searches=1)

        summary = tracker.get_summary()

        assert "total_cost" in summary
        assert "num_calls" in summary
        assert "breakdown" in summary
        assert summary["num_calls"] == 2

    def test_reset_tracker(self):
        """Test resetting cost tracker."""
        tracker = CostTracker()

        tracker.track_openai_call(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        tracker.track_tavily_search(num_searches=1)

        tracker.reset()

        assert tracker.total_cost == 0.0
        assert len(tracker.call_history) == 0

    def test_zero_tokens_handling(self):
        """Test handling of zero tokens."""
        tracker = CostTracker()

        cost = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=0,
            output_tokens=0
        )

        assert cost == 0.0
        assert tracker.total_cost == 0.0

    def test_call_history_contains_metadata(self):
        """Test that call history contains timestamp and metadata."""
        tracker = CostTracker()

        tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50
        )

        call = tracker.call_history[0]
        assert "timestamp" in call
        assert "service" in call
        assert "model" in call
        assert "cost" in call
        assert "input_tokens" in call
        assert "output_tokens" in call

    def test_unknown_model_handling(self):
        """Test handling of unknown model."""
        tracker = CostTracker()

        # Should either use default pricing or raise error
        # Depending on implementation
        try:
            cost = tracker.track_openai_call(
                model="unknown-model",
                input_tokens=100,
                output_tokens=50
            )
            # If it doesn't error, should still track something
            assert cost >= 0
        except ValueError:
            # Or it might raise an error for unknown models
            pass

    def test_export_to_dict(self):
        """Test exporting call history to dictionary."""
        tracker = CostTracker()

        tracker.track_openai_call(model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        tracker.track_tavily_search(num_searches=1)

        export = tracker.to_dict()

        assert "total_cost" in export
        assert "call_history" in export
        assert len(export["call_history"]) == 2

    def test_cost_precision(self):
        """Test that costs are calculated with appropriate precision."""
        tracker = CostTracker()

        cost = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=1,
            output_tokens=1
        )

        # Cost should be very small but positive
        assert cost > 0
        assert cost < 0.01  # Less than a cent

    def test_large_token_counts(self):
        """Test handling of large token counts."""
        tracker = CostTracker()

        cost = tracker.track_openai_call(
            model="gpt-4o-mini",
            input_tokens=100000,
            output_tokens=50000
        )

        assert cost > 0
        assert tracker.total_cost == cost
