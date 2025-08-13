"""
example of tagging text with an LLM
"""

import os
from dotenv import load_dotenv, find_dotenv

# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# the following modules are useful only if you run code in Notebooks
from IPython.display import display
from IPython.display import Markdown

# load env variables from .env file
_ = load_dotenv(find_dotenv())  # read local .env file
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive is the text on a scale of 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


# create instance of the text model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.0,
).with_structured_output(Classification)


inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
response = model.invoke(prompt)

print(response)

inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})
response = model.invoke(prompt)

print(response.dict())


print("-" * 80)


# --------------------------------------------------------------------------
# WARNING: the part below does not seem to work!
# I can't understand the error message spewed out by the interpreter :(
# --------------------------------------------------------------------------

# here is an example of controlling finer aspects of the output
from typing import Literal


# class Classification2(BaseModel):
#     sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
#     aggressiveness: int = Field(
#         ...,
#         description="describes how aggressive the statement is, the higher the number the more aggressive",
#         enum=[1, 2, 3, 4, 5],
#     )
#     language: str = Field(
#         ..., enum=["spanish", "english", "french", "german", "italian"]
#     )


class Classification2(BaseModel):
    sentiment: Literal["happy", "neutral", "sad"] = Field(
        ..., description="The sentiment of the text"
    )
    aggressiveness: Literal[1, 2, 3, 4, 5] = Field(
        ..., description="Describes how aggressive the statement is"
    )
    language: Literal["spanish", "english", "french", "german", "italian"] = Field(
        ..., description="The language the text is written in"
    )


tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.0,
).with_structured_output(Classification2)

inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
model.invoke(prompt)

print(response.dict())
