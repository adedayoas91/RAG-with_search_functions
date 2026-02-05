import os
from langchain_community.utilities import (WikipediaAPIWrapper, 
                                            TavilySearchResults, 
                                            SerperDevTool, 
                                            GoogleSearchAPIWrapper,
                                            YouTubeSearchRun)

from langchain.tools import Tool
from typing import List
from dotenv import load_dotenv
from langchain.tools import tool
from utils import logger

load_dotenv()

@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information about a given topic
    Args:
        query (str): The query to search for
    Returns:
        str: The search results
    """
    try:
        wikipedia = WikipediaAPIWrapper()
        return wikipedia.run(query)
    except Exception as e:
        logger.error(f"Error searching Wikipedia: {e}")
        return f"Error searching Wikipedia: {e}"

@tool
def search_tavily(query: str) -> str:
    """
    Use Tavily to search the web for information about a given topic
    Args:
        query (str): The query to search for
    Returns:
        str: The search results
    """
    try:
        tavily = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
        return tavily.run(query)
    except Exception as e:
        logger.error(f"Error searching Tavily: {e}")
        return f"Error searching Tavily: {e}"


@tool
def search_serper(query: str) -> str:
    """
    Use Serper to search the web for information about a given topic
    Args:
        query (str): The query to search for
    Returns:
        str: The search results
    """
    try:
        serper = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))
        return serper.run(query)
    except Exception as e:
        logger.error(f"Error searching Serper: {e}")
        return f"Error searching Serper: {e}"

@tool
def search_youtube(query: str) -> str:
    """
    Search YouTube for related videos about a given topic
    Args:
        query (str): The query to search for
    Returns:
        str: The search results
    """
    try:
        youtube = YouTubeSearchRun()
        return youtube.run(query)
    except Exception as e:
        logger.error(f"Error searching YouTube: {e}")
        return f"Error searching YouTube: {e}"

@tool
def search_google(query: str) -> str:
    """
    Use Google to search the web for information about a given topic
    Args:
        query (str): The query to search for
    Returns:
        str: The search results
    """
    try:
        google = GoogleSearchAPIWrapper()
        return google.run(query)
    except Exception as e:
        logger.error(f"Error searching Google: {e}")
        return f"Error searching Google: {e}"
