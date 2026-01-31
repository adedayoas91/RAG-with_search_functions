"""
Data persistence utilities for analytics and session data.

This module provides functions to save and load analytics data to/from JSON files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SessionData:
    """Data for a single RAG session."""

    session_id: str
    query: str
    timestamp: str
    sources_found: int
    sources_approved: int
    sources_processed: int
    chunks_created: int
    answer_length: int
    total_cost: float
    models_used: Dict[str, int]
    duration: float
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionData':
        """Create from dictionary."""
        return cls(**data)


def save_analytics(
    data: Dict,
    file_path: Optional[str] = None
) -> bool:
    """
    Save analytics data to JSON file.

    Args:
        data: Analytics data dictionary
        file_path: Path to analytics file (default: from config)

    Returns:
        True if successful, False otherwise
    """
    config = get_config()
    file_path = Path(file_path or config.paths.analytics_file)

    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Analytics saved to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save analytics: {str(e)}")
        return False


def load_analytics(
    file_path: Optional[str] = None
) -> Dict:
    """
    Load analytics data from JSON file.

    Args:
        file_path: Path to analytics file (default: from config)

    Returns:
        Analytics data dictionary (empty dict if file doesn't exist)
    """
    config = get_config()
    file_path = Path(file_path or config.paths.analytics_file)

    if not file_path.exists():
        logger.debug(f"Analytics file not found: {file_path}")
        return {
            "sessions": [],
            "total_queries": 0,
            "total_sources_analyzed": 0,
            "total_sources_processed": 0,
            "total_chunks_created": 0,
            "total_cost": 0.0,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        logger.debug(f"Analytics loaded from {file_path}")
        return data

    except Exception as e:
        logger.error(f"Failed to load analytics: {str(e)}")
        return {
            "sessions": [],
            "total_queries": 0,
            "total_sources_analyzed": 0,
            "total_sources_processed": 0,
            "total_chunks_created": 0,
            "total_cost": 0.0,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }


def update_analytics(session_data: SessionData) -> bool:
    """
    Update analytics with new session data.

    Args:
        session_data: Session data to add

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load existing analytics
        analytics = load_analytics()

        # Add new session
        analytics["sessions"].append(session_data.to_dict())

        # Update totals
        analytics["total_queries"] += 1
        analytics["total_sources_analyzed"] += session_data.sources_found
        analytics["total_sources_processed"] += session_data.sources_processed
        analytics["total_chunks_created"] += session_data.chunks_created
        analytics["total_cost"] += session_data.total_cost
        analytics["last_updated"] = datetime.now().isoformat()

        # Save back to file
        success = save_analytics(analytics)

        if success:
            logger.info(
                f"Analytics updated: Query #{analytics['total_queries']} | "
                f"Cost: ${session_data.total_cost:.4f}"
            )

        return success

    except Exception as e:
        logger.error(f"Failed to update analytics: {str(e)}")
        return False


def get_recent_sessions(
    limit: int = 10,
    file_path: Optional[str] = None
) -> List[SessionData]:
    """
    Get most recent sessions.

    Args:
        limit: Maximum number of sessions to return
        file_path: Path to analytics file (default: from config)

    Returns:
        List of SessionData objects
    """
    analytics = load_analytics(file_path)
    sessions = analytics.get("sessions", [])

    # Get last N sessions
    recent = sessions[-limit:] if len(sessions) > limit else sessions

    # Convert to SessionData objects
    return [SessionData.from_dict(s) for s in reversed(recent)]


def get_cost_summary(
    days: Optional[int] = None,
    file_path: Optional[str] = None
) -> Dict:
    """
    Get cost summary statistics.

    Args:
        days: Number of days to look back (None for all time)
        file_path: Path to analytics file (default: from config)

    Returns:
        Dictionary with cost statistics
    """
    analytics = load_analytics(file_path)
    sessions = analytics.get("sessions", [])

    # Filter by date if specified
    if days is not None:
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        sessions = [
            s for s in sessions
            if datetime.fromisoformat(s["timestamp"]).timestamp() >= cutoff_date
        ]

    if not sessions:
        return {
            "total_cost": 0.0,
            "avg_cost_per_query": 0.0,
            "num_queries": 0,
            "by_model": {},
            "period": f"last {days} days" if days else "all time"
        }

    # Calculate statistics
    total_cost = sum(s["total_cost"] for s in sessions)
    num_queries = len(sessions)
    avg_cost = total_cost / num_queries if num_queries > 0 else 0.0

    # Aggregate by model
    by_model = {}
    for session in sessions:
        for model, count in session.get("models_used", {}).items():
            if model not in by_model:
                by_model[model] = {"calls": 0, "cost": 0.0}
            by_model[model]["calls"] += count

    return {
        "total_cost": total_cost,
        "avg_cost_per_query": avg_cost,
        "num_queries": num_queries,
        "by_model": by_model,
        "period": f"last {days} days" if days else "all time"
    }


def get_source_statistics(
    file_path: Optional[str] = None
) -> Dict:
    """
    Get statistics about sources processed.

    Args:
        file_path: Path to analytics file (default: from config)

    Returns:
        Dictionary with source statistics
    """
    analytics = load_analytics(file_path)
    sessions = analytics.get("sessions", [])

    if not sessions:
        return {
            "total_sources_found": 0,
            "total_sources_approved": 0,
            "total_sources_processed": 0,
            "approval_rate": 0.0,
            "avg_sources_per_query": 0.0
        }

    total_found = sum(s["sources_found"] for s in sessions)
    total_approved = sum(s["sources_approved"] for s in sessions)
    total_processed = sum(s["sources_processed"] for s in sessions)
    num_queries = len(sessions)

    approval_rate = (
        (total_approved / total_found * 100) if total_found > 0 else 0.0
    )
    avg_per_query = total_processed / num_queries if num_queries > 0 else 0.0

    return {
        "total_sources_found": total_found,
        "total_sources_approved": total_approved,
        "total_sources_processed": total_processed,
        "approval_rate": approval_rate,
        "avg_sources_per_query": avg_per_query
    }


def export_analytics_csv(
    output_file: str,
    analytics_file: Optional[str] = None
) -> bool:
    """
    Export analytics to CSV file.

    Args:
        output_file: Path to output CSV file
        analytics_file: Path to analytics JSON file (default: from config)

    Returns:
        True if successful, False otherwise
    """
    try:
        import csv

        analytics = load_analytics(analytics_file)
        sessions = analytics.get("sessions", [])

        if not sessions:
            logger.warning("No sessions to export")
            return False

        # Write to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            # Define fieldnames
            fieldnames = [
                "session_id",
                "timestamp",
                "query",
                "sources_found",
                "sources_approved",
                "sources_processed",
                "chunks_created",
                "answer_length",
                "total_cost",
                "duration",
                "success"
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for session in sessions:
                # Extract only the fields we want
                row = {k: session.get(k, "") for k in fieldnames}
                writer.writerow(row)

        logger.info(f"Analytics exported to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export analytics to CSV: {str(e)}")
        return False


def clear_old_sessions(
    days_to_keep: int = 90,
    file_path: Optional[str] = None
) -> bool:
    """
    Remove sessions older than specified number of days.

    Args:
        days_to_keep: Number of days of history to keep
        file_path: Path to analytics file (default: from config)

    Returns:
        True if successful, False otherwise
    """
    try:
        analytics = load_analytics(file_path)
        sessions = analytics.get("sessions", [])

        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)

        # Filter sessions
        kept_sessions = [
            s for s in sessions
            if datetime.fromisoformat(s["timestamp"]).timestamp() >= cutoff_date
        ]

        removed_count = len(sessions) - len(kept_sessions)

        if removed_count > 0:
            analytics["sessions"] = kept_sessions
            success = save_analytics(analytics, file_path)

            if success:
                logger.info(f"Removed {removed_count} old sessions")

            return success
        else:
            logger.info("No old sessions to remove")
            return True

    except Exception as e:
        logger.error(f"Failed to clear old sessions: {str(e)}")
        return False
