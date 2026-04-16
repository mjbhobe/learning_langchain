"""personal_checf.py: example of using an LLM as a personal
chef to suggest recipes based on ingredients the user has on hand. This example also demonstrates how to use a tool (web search) to find recipes on the web.

NOTE: this file has been created explicitly to be run inside LangSmith studio,
so there are no "example" runs in this file and THERE IS NO NEED TO ADD Memory to agent.
"""

from dotenv import load_dotenv
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

# load API keys from local .env file
load_dotenv(override=True)

# create the web-search tool, used by our agent
tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information
    Args:
        query (str): the search query
    Returns:
        Dict[str, Any]: the search results
    """
    print(f"------ Calling web_search({query}) tool ------")
    return tavily_client.search(query)


# define our system prompt
system_prompt = """
You ar an accomplished personal chef. The user will give you a set of ingredients they have 
left over in their house. Using the web_search tool, search the web for recipies using the
ingredients they have.
Return the recipe suggestions and eventually the recipe instructions if requested.
All recipe instruction must be returned as markdown step-by-step bulleted points. Assume that
the user is new to cooking, so instructions should be as detailed as possible.
"""

# define our agent
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-5-nano",
    system_prompt=system_prompt,
    tools=[web_search],
)
