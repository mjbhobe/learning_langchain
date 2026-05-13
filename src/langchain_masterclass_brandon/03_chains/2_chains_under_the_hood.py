"""
chain_under_the_hood.py - a look at RunnableLambda & RunnableSequence
    to understand how chains work under the hood

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence

# load all environment variables
load_dotenv(override=True)

# create my LLM
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
# NOTE: a RunnableLambda is a utility function from LangChain that help you
# "convert" a regylar Python function (or a lambda) into a Runnable.
# You can chain ONLY runnables together into a chain, which you can then "invoke()"
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)
upper_case_output = RunnableLambda(lambda x: x.upper())

# and this is what LangChain does under the hood
# first & last are single items, middle can be a list of items (many items)
chain = RunnableSequence(
    first=format_prompt, middle=[invoke_model, parse_output], last=upper_case_output
)
response = chain.invoke({"topic": "Rahul Gandhi", "joke_count": "3"})
console.print(f"[green] AI Response [/green]\n {response}")
