"""
Utility modules for Agentic RAG System.

This package provides logging, cost tracking, data persistence, and CLI display utilities.
"""

from .logging_config import (
    setup_logging,
    get_logger,
    log_performance,
    log_api_call,
    log_exception,
    log_session_start,
    log_session_end
)

from .cost_tracker import (
    CostTracker,
    APICall
)

from .data_persistence import (
    SessionData,
    save_analytics,
    load_analytics,
    update_analytics,
    get_recent_sessions,
    get_cost_summary,
    get_source_statistics,
    export_analytics_csv,
    clear_old_sessions
)

from .cli_display import (
    display_sources_table,
    prompt_source_approval,
    parse_selection,
    display_progress,
    print_answer,
    print_session_summary,
    print_error,
    print_warning,
    print_success,
    confirm_action,
    display_spinner,
    clear_screen,
    print_header,
    print_divider
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "log_performance",
    "log_api_call",
    "log_exception",
    "log_session_start",
    "log_session_end",
    # Cost tracking
    "CostTracker",
    "APICall",
    # Data persistence
    "SessionData",
    "save_analytics",
    "load_analytics",
    "update_analytics",
    "get_recent_sessions",
    "get_cost_summary",
    "get_source_statistics",
    "export_analytics_csv",
    "clear_old_sessions",
    # CLI display
    "display_sources_table",
    "prompt_source_approval",
    "parse_selection",
    "display_progress",
    "print_answer",
    "print_session_summary",
    "print_error",
    "print_warning",
    "print_success",
    "confirm_action",
    "display_spinner",
    "clear_screen",
    "print_header",
    "print_divider",
]
