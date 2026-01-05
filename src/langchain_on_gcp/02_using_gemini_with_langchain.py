import os
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_google_genai import ChatGoogleGenerativeAI


# load API keys from .env file
load_dotenv(override=True)
console = Console()

# create instance of model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
)

# ask the model a question
# response = llm.invoke("Tell me how I can use Gemini 3.0 Flash in a LangChain application")
# console.print(Markdown(response.content))

# streaming
console.print("[yellow]A poem from Gemini[/yellow]\n")
prompt = "Write me a poem about Google Gemini and LangChain. Respond in Markdown format"
poem = ""
for chunk in llm.stream(prompt):
    print(chunk)