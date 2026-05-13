"""
chains_extended.py - using RunnableLambdas to extend chains

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence

# load all environment variables
load_dotenv(override=True)

# create my LLM - using Google Gemini
model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# only for colorful text & markdown output support
console = Console()

# messages for PromptTemplate (NOTE: SystemMessage must be first in the list
messages = [
    ("system", "You are a standup comedian who tells hilarious jokes on {topic}"),
    ("human", "Tell me {joke_count} jokes"),
]
prompt_template = ChatPromptTemplate.from_messages(messages)

# create our runnable lambdas
uppercase_output = RunnableLambda(lambda x: x.upper())
# count_words lambda prints count of words in x and then x on the next line
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# now build a chain
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words
# and invoke it
response = chain.invoke({"topic": "Rahul Gandhi", "joke_count": "1"})
console.print(f"[green]Chain output:[/green]\n{response}")
