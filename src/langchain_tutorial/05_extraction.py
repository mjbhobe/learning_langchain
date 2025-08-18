"""
05_extraction.py - extract structured data from text and other unstructured
    media using chat models and few-shot examples.

@author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from typing import Optional, List
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# load API keys from .env files
load_dotenv(override=True)
# for colorful text output
console = Console()

# create our LLM - we'll be using Gemini-2.5-flash
llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.0)

# in this example, we will use tool-calling features of chat models to
# extract structured information from unstructured text. We will also demonstrate
# how to use few-shot prompting in this context to improve performance.


class Person(BaseModel):
    """Information about a person."""

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(
        default=None,
        description="The name of the person",
    )
    hair_color: Optional[str] = Field(
        default=None,
        description="The color of the person's hair if known",
    )
    height_in_meters: Optional[str] = Field(
        default=None,
        description="Height measured in meters",
    )


class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

# extract 1 structured data element from text
structured_llm = llm.with_structured_output(schema=Person)

text = "Alan Smith is 6 feet tall and has blond hair."
prompt = prompt_template.invoke({"text": text})
response = structured_llm.invoke(prompt)
console.print(response)

# extract multiple structured data elements from text
# NOTE: I have a different instance of structured LLM, which is bound to a different Pydantic class
structured_llm2 = llm.with_structured_output(schema=Data)
text = (
    "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
    + "Mark is bald and he is 3/4th Jeff's height"
)
prompt = prompt_template.invoke({"text": text})
response = structured_llm2.invoke(prompt)
console.print(response)
