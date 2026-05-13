"""
prompt_template_basic.py - basic prompt template with LangChain

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

# load all environment variables
load_dotenv()

# only for colorful text & markdown output support
console = Console()

# all key LangChain classes - models, prompts, chains etc. inherit from
# Runnable class. So each sub-class implements an invoke() method

# create out prompt template with our variables - topic in this case
prompt_str = "Tell me a joke about [{topic}]"
prompt_template = ChatPromptTemplate.from_template(prompt_str)
# the invoke() call on prompt templates "formats" our message - like
# a Python format() call on a string, which replaces place holders with
# actual values of placeholders
prompt = prompt_template.invoke({"topic": "Rahul Gandhi"})
console.print(f"Prompt (with 1 param): [yellow]{prompt}[/yellow]")

# you can have multiple variables in a PromptTemplate
prompt_str = "Tell me a {type_of_joke} joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(prompt_str)
prompt = prompt_template.invoke({"type_of_joke": "political", "topic": "Rahul Gandhi"})
console.print(f"Prompt (with multiple params): [green]{prompt}[/green]")

# if you want to use Prompt templates & define message types
messages = [
    ("system", "You are a stand-up comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_count} jokes"),
]
# NOTE: instead of "from_template", you now call "from_messages"
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "Rahul Gandhi", "joke_count": "3"})
console.print(f"Prompt (with templates & message types): [blue]{prompt}[/blue]")
