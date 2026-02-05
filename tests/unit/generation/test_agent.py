
import unittest
from unittest.mock import patch, MagicMock
from src.generation.agent import create_agent
from config import get_config

class TestAgent(unittest.TestCase):
    @patch('src.generation.agent.create_langchain_agent')
    @patch('src.generation.agent.ChatOpenAI')
    @patch('src.generation.agent.get_tools')
    def test_create_agent(self, mock_get_tools, mock_chat_openai, mock_create_agent):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_tools = [MagicMock()]
        mock_get_tools.return_value = mock_tools
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        config = get_config()
        agent = create_agent(config)

        self.assertIsNotNone(agent)
        self.assertEqual(agent, mock_agent)
        mock_get_tools.assert_called_once_with(config)
        mock_chat_openai.assert_called_once_with(
            model=config.model.generation_model,
            temperature=0,
        )
        mock_create_agent.assert_called_once_with(mock_llm, mock_tools)

if __name__ == '__main__':
    unittest.main()
