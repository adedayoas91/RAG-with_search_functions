"""
YouTube transcript extraction for RAG ingestion.

This module provides functions to extract transcripts from YouTube videos
and convert them to LangChain Document format.
"""

import re
from typing import Optional
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def get_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL.

    Args:
        url: YouTube video URL

    Returns:
        Video ID if found, None otherwise

    Examples:
        >>> get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    # Support multiple YouTube URL formats
    patterns = [
        r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',  # Standard format
        r'youtu\.be/([a-zA-Z0-9_-]{11})',              # Short format
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',     # Embed format
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    logger.warning(f"Could not extract video ID from URL: {url}")
    return None

def get_transcript(url: str) -> Optional[list]:
    """
    Extract English transcript from YouTube video.

    Prioritizes manually created transcripts over auto-generated ones.

    Args:
        url: YouTube video URL

    Returns:
        List of transcript segments (dicts with 'text', 'start', 'duration')
        or None if no transcript found

    Raises:
        Exception: If video ID is invalid or transcript unavailable
    """
    video_id = get_video_id(url)

    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {url}")

    try:
        # Fetch the list of available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript = None

        # Try to get manually created English transcript first
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
            logger.info(f"Found manually created English transcript for {video_id}")
        except:
            # Fall back to auto-generated English transcript
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                logger.info(f"Found auto-generated English transcript for {video_id}")
            except:
                logger.warning(f"No English transcript available for {video_id}")
                return None

        if transcript:
            return transcript.fetch()

        return None

    except Exception as e:
        logger.error(f"Error fetching transcript for {video_id}: {str(e)}")
        raise

def process(transcript: list) -> str:
    """
    Process transcript segments into a formatted string.

    Args:
        transcript: List of transcript segments (dicts with 'text', 'start', 'duration')

    Returns:
        Formatted transcript string with timestamps
    """
    if not transcript:
        return ""

    # Initialize an empty string to hold the formatted transcript
    txt = ""

    # Loop through each entry in the transcript
    for segment in transcript:
        try:
            # Access dictionary keys (not attributes)
            text = segment['text']
            start = segment['start']

            # Format timestamp as MM:SS
            minutes = int(start // 60)
            seconds = int(start % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            # Append the text and its start time to the output string
            txt += f"[{timestamp}] {text}\n"

        except (KeyError, TypeError) as e:
            # If there is an issue accessing keys, skip this entry
            logger.warning(f"Skipping malformed transcript segment: {e}")
            continue

    return txt.strip()


def load_youtube_video(url: str) -> Document:
    """
    Load YouTube video transcript as a LangChain Document.

    This is the main function to use for loading YouTube videos into the RAG system.

    Args:
        url: YouTube video URL

    Returns:
        LangChain Document with transcript text and metadata

    Raises:
        ValueError: If URL is invalid
        Exception: If transcript cannot be fetched

    Example:
        >>> doc = load_youtube_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        >>> print(doc.page_content[:100])
        >>> print(doc.metadata)
    """
    logger.info(f"Loading YouTube video: {url}")

    try:
        # Get transcript
        transcript = get_transcript(url)

        if not transcript:
            raise ValueError(f"No transcript available for video: {url}")

        # Process transcript into formatted text
        text = process(transcript)

        if not text:
            raise ValueError(f"Empty transcript for video: {url}")

        # Extract video ID for metadata
        video_id = get_video_id(url)

        # Create Document with metadata
        doc = Document(
            page_content=text,
            metadata={
                "source": url,
                "source_type": "youtube",
                "video_id": video_id,
                "transcript_length": len(text),
                "num_segments": len(transcript)
            }
        )

        logger.info(
            f"Successfully loaded YouTube video {video_id}: "
            f"{len(text)} characters, {len(transcript)} segments"
        )

        return doc

    except Exception as e:
        logger.error(f"Failed to load YouTube video {url}: {str(e)}")
        raise