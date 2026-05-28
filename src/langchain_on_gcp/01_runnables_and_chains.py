"""
runnables_and_chains.py - exploring LangChain's runnables & chains

@Author: Manish Bhobe
My experiments with LangChain on the GCP. Code is shared for learning purposes only!
"""

import random
from operator import itemgetter
from rich.console import Console
import langchain
from langchain_core.runnables import RunnableLambda, RunnableParallel

# SEED = 41
# random.seed(SEED)
console = Console()  # for colorful text on console

console.print(f"[red]Using Langchain:[/red] {langchain.__version__}")


# base class for all LangChain classes is Runnable, which has an invoke() method
# apart from others such as stream() and batch()
# A chain is formed from Runnables "chained" together and itself is a Runnable
# A RunnableLambda is a class that makes any Python function/lambda a Runnable object

runnable = RunnableLambda(lambda x: x + 1)
console.print(f"[sky_blue1]runnable.invoke(1) = [/sky_blue1] {runnable.invoke(1)}")
console.print(f"[sky_blue1]runnable.invoke(2) = [/sky_blue1] {runnable.invoke(2)}")


# runnables can be chained together into a chain, which itself is a runnable
def get_radius() -> int:
    """return a random number between 1 & 15"""
    radius = random.randint(1, 15)
    console.print(f"[red]Generated radius: {radius}[/red]")
    return radius


def calculate_circle_area(radius: int) -> str:
    """calculate the area of a circle given its radius"""
    return f"Area of circle with radius {radius} is {3.14 * radius * radius:.3f}"


def fake_llm(x: int) -> str:
    return f"Fake LLM says: {x}^2 = {x**2}"


# now build a chain like this using LCEL (LangChain Expression Language)
chain = (
    # have used the funny lambda syntax for the first RunnableLambda as
    # get_radius() does not take a parameter, but invoke() requires at least 1
    RunnableLambda(lambda _: get_radius())
    | RunnableLambda(calculate_circle_area)
)

# think of the above chain as expanding in reverse order as below:
# calculate_circle_area_runnable_lambda.invoke(get_radius_runnable_lambda.invoke(None))


# NOTE: the invoke() function requires a parameter!
response = chain.invoke(None)
console.print(f"[sky_blue1]Response of chain ->[/sky_blue1] {response}")

# you can also run chains in parallel
parallel_chain = RunnableParallel(
    step1=chain, step2=RunnableLambda(fake_llm)  # same chain as above
)
response = parallel_chain.invoke(7)
console.print(response)


# I have invoked the parallel chain with a single number
# Normally we would pass a dictionary to the invoke() function
# In that case, to extract value of key we need to pre-pend itemgetter to the head of our chain
itemgetter_chain = itemgetter("x") | RunnableLambda(fake_llm)
response = itemgetter_chain.invoke({"x": 20})
console.print(response)
