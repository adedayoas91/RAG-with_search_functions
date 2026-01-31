"""
CLI display utilities for source approval and result presentation.

This module provides formatted tables and interactive prompts for the CLI interface.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def display_sources_table(summaries: List) -> None:
    """
    Display sources in a formatted table.

    Args:
        summaries: List of SourceSummary objects
    """
    if not summaries:
        print("\nNo sources to display.")
        return

    print("\n" + "=" * 120)
    print("SOURCES FOUND")
    print("=" * 120)

    # Header
    print(f"{'#':<4} {'Title':<40} {'Type':<10} {'Score':<8} {'Summary':<50}")
    print("-" * 120)

    # Rows
    for idx, summary in enumerate(summaries, 1):
        source = summary.source
        title = (source.title[:37] + "...") if len(source.title) > 40 else source.title
        source_type = source.source_type
        score = f"{summary.relevance_score:.2f}"
        summary_text = (summary.summary[:47] + "...") if len(summary.summary) > 50 else summary.summary

        print(f"{idx:<4} {title:<40} {source_type:<10} {score:<8} {summary_text:<50}")

    print("=" * 120)
    print(f"\nTotal: {len(summaries)} sources")


def prompt_source_approval(summaries: List) -> List:
    """
    Show sources table and prompt user to approve/reject.

    Args:
        summaries: List of SourceSummary objects

    Returns:
        List of approved SearchResult objects
    """
    if not summaries:
        print("\nNo sources available for approval.")
        return []

    # Display the table
    display_sources_table(summaries)

    print("\n" + "=" * 120)
    print("SOURCE APPROVAL")
    print("=" * 120)
    print("Options:")
    print("  - 'all' or 'a'  : Approve all sources")
    print("  - 'none' or 'n' : Reject all sources")
    print("  - '1,3,5'       : Approve specific sources (comma-separated)")
    print("  - '1-5'         : Approve range of sources")
    print("  - Mix: '1,3-5,7': Approve 1, 3 through 5, and 7")
    print("=" * 120)

    while True:
        try:
            selection = input("\nApprove sources: ").strip().lower()

            if not selection:
                print("Please enter a selection.")
                continue

            # Handle 'all' or 'a'
            if selection in ['all', 'a']:
                approved = [s.source for s in summaries]
                print(f"\nApproved: All {len(approved)} sources")
                return approved

            # Handle 'none' or 'n'
            if selection in ['none', 'n']:
                print("\nApproved: 0 sources")
                return []

            # Parse selection
            indices = parse_selection(selection, len(summaries))

            if not indices:
                print("No valid sources selected. Please try again.")
                continue

            # Get approved sources
            approved = [summaries[i - 1].source for i in indices]
            print(f"\nApproved: {len(approved)} source(s) - #{', #'.join(map(str, sorted(indices)))}")

            return approved

        except KeyboardInterrupt:
            print("\n\nSelection cancelled.")
            return []
        except Exception as e:
            logger.error(f"Error in source approval: {str(e)}")
            print(f"Error: {str(e)}. Please try again.")


def parse_selection(selection: str, max_index: int) -> List[int]:
    """
    Parse user selection string into list of indices.

    Args:
        selection: Selection string (e.g., '1,3,5' or '1-5' or '1,3-5,7')
        max_index: Maximum valid index

    Returns:
        List of selected indices (1-based)

    Examples:
        parse_selection('1,3,5', 10) -> [1, 3, 5]
        parse_selection('1-5', 10) -> [1, 2, 3, 4, 5]
        parse_selection('1,3-5,7', 10) -> [1, 3, 4, 5, 7]
    """
    indices = set()

    # Split by comma
    parts = selection.split(',')

    for part in parts:
        part = part.strip()

        # Handle range (e.g., '1-5')
        if '-' in part:
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())

                if start < 1 or end > max_index or start > end:
                    logger.warning(
                        f"Invalid range: {part} (valid range: 1-{max_index})"
                    )
                    continue

                indices.update(range(start, end + 1))

            except ValueError:
                logger.warning(f"Invalid range format: {part}")
                continue

        # Handle single number
        else:
            try:
                num = int(part)

                if num < 1 or num > max_index:
                    logger.warning(
                        f"Invalid index: {num} (valid range: 1-{max_index})"
                    )
                    continue

                indices.add(num)

            except ValueError:
                logger.warning(f"Invalid number: {part}")
                continue

    return sorted(list(indices))


def display_progress(message: str, step: int = 0, total: int = 0) -> None:
    """
    Display progress message.

    Args:
        message: Progress message
        step: Current step (optional)
        total: Total steps (optional)
    """
    if step > 0 and total > 0:
        percentage = (step / total) * 100
        print(f"\n[{percentage:5.1f}%] {message} ({step}/{total})")
    else:
        print(f"\n{message}")


def print_answer(answer) -> None:
    """
    Format and display final RAG answer.

    Args:
        answer: GeneratedAnswer object
    """
    print("\n" + "=" * 120)
    print("ANSWER")
    print("=" * 120)
    print(f"\n{answer.answer}\n")

    if answer.sources:
        print("-" * 120)
        print("SOURCES CITED:")
        for idx, source in enumerate(answer.sources, 1):
            print(f"  [{idx}] {source}")

    print("\n" + "=" * 120)
    print("METADATA")
    print("=" * 120)
    print(f"Model: {answer.model}")
    print(f"Tokens Used: {answer.tokens_used:,}")
    print(f"Cost: ${answer.cost:.4f}")
    print("=" * 120)


def print_session_summary(
    sources_found: int,
    sources_approved: int,
    sources_processed: int,
    chunks_created: int,
    total_cost: float,
    duration: float
) -> None:
    """
    Print summary of the session.

    Args:
        sources_found: Number of sources found
        sources_approved: Number of sources approved by user
        sources_processed: Number of sources successfully processed
        chunks_created: Number of chunks created
        total_cost: Total cost in USD
        duration: Session duration in seconds
    """
    print("\n" + "=" * 120)
    print("SESSION SUMMARY")
    print("=" * 120)
    print(f"Sources Found:      {sources_found}")
    print(f"Sources Approved:   {sources_approved}")
    print(f"Sources Processed:  {sources_processed}")
    print(f"Chunks Created:     {chunks_created:,}")
    print(f"Total Cost:         ${total_cost:.4f}")
    print(f"Duration:           {duration:.1f}s")
    print("=" * 120)


def print_error(message: str, details: Optional[str] = None) -> None:
    """
    Print formatted error message.

    Args:
        message: Error message
        details: Additional error details (optional)
    """
    print("\n" + "=" * 120)
    print("ERROR")
    print("=" * 120)
    print(f"\n{message}\n")

    if details:
        print(f"Details: {details}\n")

    print("=" * 120)


def print_warning(message: str) -> None:
    """
    Print formatted warning message.

    Args:
        message: Warning message
    """
    print(f"\n⚠️  WARNING: {message}")


def print_success(message: str) -> None:
    """
    Print formatted success message.

    Args:
        message: Success message
    """
    print(f"\n✓ {message}")


def confirm_action(prompt: str, default: bool = False) -> bool:
    """
    Ask user for yes/no confirmation.

    Args:
        prompt: Confirmation prompt
        default: Default value if user just presses enter

    Returns:
        True if user confirmed, False otherwise
    """
    suffix = " [Y/n]" if default else " [y/N]"
    full_prompt = prompt + suffix + ": "

    while True:
        try:
            response = input(full_prompt).strip().lower()

            if not response:
                return default

            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please answer 'y' or 'n'.")

        except KeyboardInterrupt:
            print("\n")
            return False


def display_spinner(message: str = "Processing...") -> None:
    """
    Display a simple spinner animation (placeholder).

    Args:
        message: Message to display with spinner

    Note:
        This is a simple placeholder. For a real spinner, consider using
        libraries like 'halo' or 'yaspin'.
    """
    print(f"\n{message}")


def clear_screen() -> None:
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str) -> None:
    """
    Print a formatted header.

    Args:
        title: Header title
    """
    width = 120
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_divider() -> None:
    """Print a divider line."""
    print("-" * 120)
