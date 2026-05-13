"""
1_agents_and_tools_basics.py - basic agent with time tool

@author: Manish Bhobé
My experiments with Python, AI/ML and Gen AI.
Code is shared for learning purposed only - use at own risk!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv(override=True)
console = Console()


def get_current_time(*args, **kwargs) -> str:
    """Returns the current time in H:MM AM/PM format."""
    from datetime import datetime

    return datetime.now().strftime("%I:%M %p")


tools = [
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="Get the current time in HH:MM AM/PM format.",
    )
]

model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# create_react_agent returns a compiled LangGraph with ReAct logic built-in
agent_executor = create_react_agent(model=model, tools=tools)

response = agent_executor.invoke(
    {"messages": [{"role": "user", "content": "What is the current time?"}]}
)
console.print(f"[green]Response:[/green] {response['messages'][-1].content}")
