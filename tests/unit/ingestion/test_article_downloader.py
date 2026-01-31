"""
Unit tests for article downloading and parsing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.ingestion.article_downloader import (
    sanitize_filename,
    is_downloadable_article,
    download_article_wget,
    parse_and_save_article,
    download_articles_from_sources
)


@pytest.mark.unit
class TestArticleDownloader:
    """Tests for article downloading functionality."""

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        assert sanitize_filename("Normal Title") == "Normal_Title"
        assert sanitize_filename("Title: With Special/Characters") == "Title_With_Special_Characters"
        assert sanitize_filename("Title?!@#$%^&*()") == "Title_"
        assert sanitize_filename("a" * 200) == "a" * 150  # Truncates to 150 chars

    def test_is_downloadable_article(self):
        """Test detection of downloadable articles."""
        assert is_downloadable_article("https://example.com/paper.pdf") is True
        assert is_downloadable_article("https://arxiv.org/pdf/2101.12345.pdf") is True
        assert is_downloadable_article("https://example.com/article.html") is False
        assert is_downloadable_article("https://youtube.com/watch?v=test") is False

    @patch('subprocess.run')
    def test_download_article_wget_success(self, mock_run, temp_dir):
        """Test successful article download with wget."""
        mock_run.return_value = Mock(returncode=0)

        # Create a mock PDF file
        test_pdf = temp_dir / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 test content")

        with patch('pathlib.Path.glob', return_value=[test_pdf]):
            success, file_path = download_article_wget(
                url="https://example.com/paper.pdf",
                output_dir=temp_dir,
                timeout=30
            )

        assert success is True
        assert file_path is not None

    @patch('subprocess.run')
    def test_download_article_wget_failure(self, mock_run, temp_dir):
        """Test failed article download with wget."""
        mock_run.return_value = Mock(returncode=1)

        success, file_path = download_article_wget(
            url="https://example.com/paper.pdf",
            output_dir=temp_dir,
            timeout=30
        )

        assert success is False
        assert file_path is None

    @patch('requests.get')
    def test_parse_and_save_article_success(self, mock_get, temp_dir):
        """Test successful article HTML parsing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <body>
                <article>
                    <h1>Test Article</h1>
                    <p>This is the main content of the article.</p>
                </article>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        success, file_path = parse_and_save_article(
            url="https://example.com/article",
            output_dir=temp_dir,
            title="Test Article",
            timeout=30
        )

        assert success is True
        assert file_path is not None
        assert file_path.exists()
        assert file_path.suffix == ".txt"

        # Check content
        content = file_path.read_text()
        assert "Test Article" in content
        assert "main content" in content

    @patch('requests.get')
    def test_parse_and_save_article_failure(self, mock_get, temp_dir):
        """Test failed article HTML parsing."""
        mock_get.side_effect = Exception("Connection error")

        success, file_path = parse_and_save_article(
            url="https://example.com/article",
            output_dir=temp_dir,
            title="Test Article"
        )

        assert success is False
        assert file_path is None

    @patch('requests.get')
    def test_parse_removes_unwanted_elements(self, mock_get, temp_dir):
        """Test that parsing removes scripts, styles, nav, footer."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <body>
                <nav>Navigation</nav>
                <script>alert('test');</script>
                <article>
                    <p>Main content</p>
                </article>
                <footer>Footer content</footer>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        success, file_path = parse_and_save_article(
            url="https://example.com/article",
            output_dir=temp_dir,
            title="Test"
        )

        content = file_path.read_text()
        assert "Main content" in content
        assert "Navigation" not in content
        assert "alert" not in content
        assert "Footer content" not in content

    @patch('src.ingestion.article_downloader._download_single_article')
    def test_download_articles_from_sources(self, mock_download, temp_dir, mock_search_results):
        """Test parallel downloading of multiple articles."""
        # Mock successful downloads
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"test")
        mock_download.return_value = (mock_search_results[0], test_file)

        saved = download_articles_from_sources(
            sources=mock_search_results[:3],
            query="test query",
            base_dir=str(temp_dir),
            min_downloads=2,
            max_workers=2
        )

        assert len(saved) >= 2
        assert all(isinstance(item, tuple) for item in saved)

    @patch('src.ingestion.article_downloader._download_single_article')
    def test_download_articles_continues_until_minimum(self, mock_download, temp_dir, mock_search_results):
        """Test that download continues until minimum is met."""
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"test")

        # First two calls return None, third succeeds
        mock_download.side_effect = [
            None,
            None,
            (mock_search_results[0], test_file),
            (mock_search_results[1], test_file),
            (mock_search_results[2], test_file)
        ]

        with patch('builtins.input', return_value='y'):
            saved = download_articles_from_sources(
                sources=mock_search_results[:5],
                query="test",
                base_dir=str(temp_dir),
                min_downloads=3,
                max_workers=1
            )

        assert len(saved) >= 3
