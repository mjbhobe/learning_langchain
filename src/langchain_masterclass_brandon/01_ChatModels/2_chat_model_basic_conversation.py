"""
chat_model_basic_conversation.py - extending the basic Q&A chat model to
    simiulate a conversation with SystemMessage, HumanMessage & AIMessage
    using LangChain.

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown

# load all environment variables
load_dotenv(override=True)

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

# setup the messages
messages = [
    SystemMessage("Solve the following problem."),
    HumanMessage("What is 81 divided by 9?"),
]

# and invoke the model with these messages
response = model.invoke(messages)
console.print(response.content)

# simulate a conversation (as you would in ChatGPT)
messages = [
    SystemMessage("Solve the following problem."),
    # question I ask
    HumanMessage("What is 81 divided by 9?"),
    # response I get from LLM
    AIMessage("81 divided by 9 is 9."),
    # next question I ask (expecting ~63.62)
    # HumanMessage("What is area of circle with radius 4.5?"),
    # since LLM has complete context, it should respond with 9 x 3 = 27 (or something similar)
    HumanMessage("What is answer of previous question multiplied by 3?"),
]

# and invoke the model with these messages
response = model.invoke(messages)
console.print(Markdown(response.content))
