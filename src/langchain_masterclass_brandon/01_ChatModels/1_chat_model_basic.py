"""
chat_model_basic.py - basic chat model structure with LangChain

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console

# load all environment variables
load_dotenv(override=True)

model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
console = Console()

question = "What is 81 divided by 9?"
console.print(f"[blue]Human: [/blue] {question}")
response = model.invoke(question)
console.print("[yellow]Entire response from model:[/yellow]")
console.print(response)
console.print(f"[blue]AI: [/blue] {response.content}")
