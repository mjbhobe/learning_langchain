"""
extracting structured data from unstructured text
"""

import os
from dotenv import load_dotenv, find_dotenv

# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# load env variables from .env file
_ = load_dotenv(find_dotenv())  # read local .env file


from typing import Optional, List
from pydantic import BaseModel, Field


class Person(BaseModel):
    """information about a Person"""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
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

# --------------------------------------------------------------
# NOTE: Google Gemini does not work very well & consistently
# for this kind of problem. OpenAI works very well and is
# able to extract all entities irrespective of order in
# which they appear in the sentence
# --------------------------------------------------------------

# define our LLM - as usual we'll be using Google Gemini
# create instance of the text model
# model = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     google_api_key=os.environ["GOOGLE_API_KEY"],
# )

# from langchain_openai import ChatOpenAI

# model = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

structured_llm = model.with_structured_output(schema=Person)

# let's see what it extracts from this text
# it has all 3 attributes - name, hair color and height (but in feet, not meters!)
text = "Alan Smith is 6 feet tall and has blond hair."
prompt = prompt_template.invoke({"text": text})
response = structured_llm.invoke(prompt)
print(response)  # will extract just namne - can't extract hair color or height

# slight change in text - let's see what happens now
text = "Alan Smith has blond hair and is 6 feet tall."
prompt = prompt_template.invoke({"text": text})
response = structured_llm.invoke(prompt)
print(response)  # can now extract both name & hair color, but not height

# and this??
text = "Alan Smith has blond hair and is 1.82 meters tall."
prompt = prompt_template.invoke({"text": text})
response = structured_llm.invoke(prompt)
print(response)  # can now extract all 3 attribs - name, hair color & height


# now we extend the example by extracting list of Person attributes
class Data(BaseModel):
    """extract data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]


# structured_llm = model.with_structured_output(schema=Data)
structured_llm = model.with_structured_output(schema=Data)
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.invoke({"text": text})
response = structured_llm.invoke(prompt)
print(response)
