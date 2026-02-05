
import unittest
from unittest.mock import patch, MagicMock
from src.generation.tools import get_tools
from config import get_config

class TestTools(unittest.TestCase):
    @patch('src.generation.tools.TavilySearchResults')
    @patch('src.generation.tools.WikipediaQueryRun')
    @patch('src.generation.tools.WikipediaAPIWrapper')
    def test_get_tools(self, mock_wikipedia_api_wrapper, mock_wikipedia_query_run, mock_tavily_search_results):
        mock_tavily = MagicMock()
        mock_wikipedia = MagicMock()
        mock_api_wrapper = MagicMock()
        mock_tavily_search_results.return_value = mock_tavily
        mock_wikipedia_query_run.return_value = mock_wikipedia
        mock_wikipedia_api_wrapper.return_value = mock_api_wrapper

        config = get_config()
        tools = get_tools(config)

        self.assertEqual(len(tools), 2)
        self.assertIn(mock_tavily, tools)
        self.assertIn(mock_wikipedia, tools)
        mock_tavily_search_results.assert_called_once_with(
            max_results=config.search.max_search_results,
        )
        mock_wikipedia_query_run.assert_called_once_with(api_wrapper=mock_api_wrapper)
        mock_wikipedia_api_wrapper.assert_called_once()

if __name__ == '__main__':
    unittest.main()
