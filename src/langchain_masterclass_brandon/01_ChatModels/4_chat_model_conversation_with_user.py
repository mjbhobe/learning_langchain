"""
chat_model_conversation_with_user.py - console based ChatGPT like application
    using chat history, SystemMessage, HumanMessage & AIMessage in Langchain

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown

# load all environment variables
load_dotenv()

# create an instance of the LLM (we'll use OpenAI GPT 5 Nano)
model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# only for colorful text & markdown output support
console = Console()

# this is our chat history
chat_history = []

# setup the messages
message = SystemMessage(
    content="""You are a helpful AI assistant that can anwser questions
    on a variety of subjects. Be brief with your responses and use a casual tone."""
)
chat_history.append(message)
console.print("""[green]Welcome to ChatterBox, the console chat application.\n[/green]
Ask your questions at the [blue]Your query?[/blue] prompt (OR type quit/exit/bye to quit ChatterBox)""")

# endless loop unless user types "quit" or "exit" or "bye"
while True:
    console.print("[blue]Your query? [/blue]", end="")
    query = input().strip().lower()
    if query in ["quit", "exit", "bye"]:
        break
    human_message = HumanMessage(content=query)
    chat_history.append(human_message)
    # console.print("[yellow]AI generating...[/yellow]", end="")

    # pass the entire history to LLM
    response = model.invoke(chat_history)
    # NOTE: append ENTIRE response, not just response.content
    chat_history.append(response)
    console.print("\r[green]AI response:[/green]")
    console.print(Markdown(response.content))

# finish up by showing the entire chat history
console.print("[red] ----------- Chat history ----------- [/red]")
console.print(chat_history)
