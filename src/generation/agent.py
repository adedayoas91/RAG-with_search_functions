
from langchain.agents import create_agent as create_langchain_agent
from langchain_openai import ChatOpenAI
from src.generation.tools import get_tools

def create_agent(config):
    """
    Creates and returns a LangChain agent using the modern LangChain 1.x API.
    """
    tools = get_tools(config)
    llm = ChatOpenAI(
        model=config.model.generation_model,
        temperature=0,
    )

    # Use the modern LangChain 1.x agent creation API
    # create_langchain_agent returns a CompiledStateGraph (Runnable) instead of AgentExecutor
    agent = create_langchain_agent(llm, tools)

    return agent
