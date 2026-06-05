"""tests the Anthropic API"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

import anthropic
from claude_client import ClaudeClient

load_dotenv(override=True)
console = Console()

USER_QUESTION = "Write a Python function that validates an email address using regexp."

client = ClaudeClient()

# # Attempt 1: call without a system prompt
# response = client.ask(
#     prompt=USER_QUESTION,
# )
# console.print(Markdown(response.content[0].text))

## Here is an example of using an EABORATE system prompt
SYSTEM_PROMPT = """
You are a senior Python engineer with 10 years of experience.
You write clean, production-ready code that follows these standards:

CODING STANDARDS:
- PEP 8 style- Full type annotations (Python 3.10+ union syntax)
- Docstrings in Google format
- Comprehensive error handling with specific exception types
- Never use bare "except:" clauses

RESPONSE FORMAT:
- Lead with the code
- Follow with a brief explanation of key decisions
- Flag any assumptions you made
- Note any edge cases the implementation doesn't handle

TECH STACK CONTEXT:
 - Python 3.11, FastAPI 0.100+, SQLAlchemy 2.0, PostgreSQL 15
 - Redis for caching, Celery for background tasks
 - pytest for testing
"""

# Attempt 1: call WITH a system prompt
response = client.ask(
    prompt=USER_QUESTION,
    system_prompt=SYSTEM_PROMPT,
)
console.print(Markdown(response.content[0].text))

#
