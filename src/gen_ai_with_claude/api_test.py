"""tests the Anthropic API"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

import anthropic
from gen_ai_with_claude import ClaudeClient

load_dotenv(override=True)
console = Console()

client = ClaudeClient()
response = client.ask(
    prompt="Write a Python function that validates an email address using regexp.",
)

console.print(Markdown(response.content[0].text))
