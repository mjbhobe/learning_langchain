"""
02_prompt_templates.py - in this example we'll create a prompt template
    that helps you translate any user input from English to any language
    user specifies.

    We'll be using Google Gemini Flash 2.x model in this series, but you
    can you any LLM, including open source LLMs, of your choice.

@author: Manish Bhobe
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)
console = Console()

# we'll use Gemini 2.0 flash
llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0.0)

system_message: str = "Translate the following from English into {language}"

# create our prompt template - note that we have 2 variables
# language & user_input to get from the user
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_message), ("user", "{user_input}")]
)
# get inputs from the user
console.print("[green]Sentence to translate: [/green]", end="")
user_input = input()
console.print("[green]Language to translate to: [/green]", end="")
language_input = input()
# build the prompt (fill in the variable values we inputted from user)
prompt = prompt_template.invoke({"language": language_input, "user_input": user_input})
console.print(f"[blue]Prompt: [/blue]{prompt}")

# call the LLM to do the translation of user inputted sentence & language choice
response = llm.invoke(prompt)
console.print(f"[yellow]AI: [/yellow]{response.content}")
