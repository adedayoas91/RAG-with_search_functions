
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def get_tools(config):
    """
    Returns a list of tools for the agent to use.
    """
    tavily_tool = TavilySearchResults(
        max_results=config.max_results,
    )

    wikipedia_tool = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(),
    )

    return [tavily_tool, wikipedia_tool]
