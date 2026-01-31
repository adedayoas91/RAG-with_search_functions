"""
Unit tests for YouTube transcript extraction.
"""

import pytest
from unittest.mock import Mock, patch
from src.ingestion.yt_bot import get_video_id, process, load_youtube_video


@pytest.mark.unit
class TestYouTubeBot:
    """Tests for YouTube transcript functionality."""

    def test_get_video_id_standard_format(self):
        """Test extracting video ID from standard YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = get_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_get_video_id_short_format(self):
        """Test extracting video ID from short YouTube URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        video_id = get_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_get_video_id_embed_format(self):
        """Test extracting video ID from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        video_id = get_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_get_video_id_invalid_url(self):
        """Test handling invalid YouTube URL."""
        url = "https://example.com/not-youtube"
        video_id = get_video_id(url)
        assert video_id is None

    def test_process_transcript(self, mock_youtube_transcript):
        """Test processing transcript into formatted string."""
        result = process(mock_youtube_transcript)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "[00:00]" in result
        assert "Welcome to this video" in result
        assert "[00:02]" in result
        assert "Today we'll discuss AI" in result

    def test_process_empty_transcript(self):
        """Test processing empty transcript."""
        result = process([])
        assert result == ""

    def test_process_malformed_segment(self):
        """Test processing transcript with malformed segments."""
        transcript = [
            {"text": "Good segment", "start": 0.0},
            {"invalid": "Bad segment"},  # Missing required keys
            {"text": "Another good one", "start": 5.0}
        ]

        result = process(transcript)
        # Should skip malformed segment but process good ones
        assert "Good segment" in result
        assert "Another good one" in result

    @patch('src.ingestion.yt_bot.get_video_id')
    @patch('src.ingestion.yt_bot.YouTubeTranscriptApi')
    def test_load_youtube_video_success(self, mock_api, mock_get_video_id, mock_youtube_transcript):
        """Test successfully loading a YouTube video."""
        # Mock video ID extraction (YouTube IDs are 11 chars, but test uses "test123")
        mock_get_video_id.return_value = "test123"

        # Mock the API
        mock_transcript_list = Mock()
        mock_transcript = Mock()
        mock_transcript.fetch.return_value = mock_youtube_transcript
        mock_transcript_list.find_manually_created_transcript.return_value = mock_transcript
        mock_api.list_transcripts.return_value = mock_transcript_list

        url = "https://www.youtube.com/watch?v=test123"
        doc = load_youtube_video(url)

        assert doc is not None
        assert len(doc.page_content) > 0
        assert doc.metadata["source"] == url
        assert doc.metadata["source_type"] == "youtube"
        assert doc.metadata["video_id"] == "test123"

    @patch('src.ingestion.yt_bot.YouTubeTranscriptApi')
    def test_load_youtube_video_no_transcript(self, mock_api):
        """Test handling video with no transcript."""
        mock_api.list_transcripts.side_effect = Exception("No transcript available")

        url = "https://www.youtube.com/watch?v=test123"

        with pytest.raises(Exception):
            load_youtube_video(url)

    def test_load_youtube_video_invalid_url(self):
        """Test handling invalid YouTube URL."""
        url = "https://example.com/not-youtube"

        with pytest.raises(ValueError):
            load_youtube_video(url)
