"""
runnables_and_chains.py - exploring LangChain's runnables & chains

@Author: Manish Bhobe
My experiments with LangChain on the GCP. Code is shared for learning purposes only!
"""
import random
from langchain_core.runnables import RunnableLambda

SEED = 41
random.seed(SEED)

# base class for all LangChain classes is Runnable, which has an invoke() method
# apart from others such as stream() and batch()
# A chain is formed from Runnables "chained" together and itself is a Runnable
# A RunnableLambda is a class that makes any Python function/lambda a Runnable object

runnable = RunnableLambda(lambda x: x + 1)
print(f"runnable.invoke(1) = {runnable.invoke(1)}")
print(f"runnable.invoke(2) = {runnable.invoke(2)}")

# runnables can be chained together into a chain, which itself is a runnable
def get_radius() -> int:
    """ return a random number between 1 & 15 """
    return random.randint(1, 15)

def calculate_circle_area(radius: int) -> float:
    """ calculate the area of a circle given its radius """
    return f"Area of circle with radius {radius} is {3.14 * radius * radius}"

# now build a chain like this using LCEL (LangChain Expression Language)
chain = (
    # have used the funny lambda syntax for the first RunnableLambda as
    # get_radius() does not take a parameter, but invoke()
    RunnableLambda(lambda _:get_radius()) | RunnableLambda(calculate_circle_area)
)

# NOTE: the invoke() function requires a parameter!
response = chain.invoke(None)
print(f"Response of chain -> {response}")
