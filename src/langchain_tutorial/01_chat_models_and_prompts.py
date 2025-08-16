"""
01_chat_models_and_prompts.py - calling an LLM with Langchain and System
    and Human messages, without prompt templated. We'll be using the
    Google Gemini Flash 2.x model in this series, but you can you any
    LLM, including open source LLMs, of your choice.

@author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(override=True)
console = Console()

# we'll use Gemini 2.0 flash
llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0.0)

system_message: str = "Translate the following from English into Italian"
human_message: str = "Wassup dude!! Welcome to LangChain"

messages = [
    SystemMessage(system_message),
    HumanMessage(human_message),
]

response = llm.invoke(messages)
console.print(f"[blue]System: [/blue]{system_message}")
console.print(f"[green]Human: [/green]{human_message}")
console.print(f"[yellow]AI: [/yellow]{response.content}")

# also try streaming the response
console.print("[red]With streaming...[/red]")
for token in llm.stream(messages):
    console.print(f"[yellow]{token.content}[/yellow]", end="|")
