"""
chain_basics.py - basics of LangChain chains.

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load all environment variables
load_dotenv()

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

# NOTE: LangChain classes, such as Prompt Templates, Models, parsers are all derived
# from Runnable base class, which makes all of them Runnable. A Runnable class has a
# common invoke() function as well as | operator. The latter works to "chain" output
# of previous Runnable to the input of next Runnable in the chain

# now chain the prompt template + model + output parser
# you can read this as send output of prompt_template.invoke() to model as input
# call model.invoke() with that input and send output as input to StrOutputParser
# send output of StrOutpurParser back to caller (as it's the final object in the chain)
chain = prompt_template | model | StrOutputParser()

# chaining Runnables together (as we have done above), also returns a Runnable
# I can call invoke() on this Runnable to kick-off the chain
params = {"topic": "Rahul Gandhi", "joke_count": "3"}
response = chain.invoke(params)  # params are parameters to first Runnable in chain!
console.print(
    f"[red] --- Showing {params['joke_count']} LLM generated jokes on {params['topic']} ---[/red]"
)
# NOTE: since I have StrOutputParser() at the end of the chain, my response
# is a string - so don't need response.content
console.print(f"[green] LLM Response [/green]: \n {response}")
