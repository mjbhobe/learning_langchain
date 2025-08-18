"""
04_classification.py - Classify text into categories or labels using
    chat models with structured outputs.

@author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# load API keys from .env files
load_dotenv(override=True)
# for colorful text output
console = Console()

# create our LLM - we'll be using Gemini-2.5-flash
llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.0)

# in this example, we'll provide the model with some text and ask it to
# sentiment, aggressiveness and language of the text
# A trick is to use a Pydantic model to define the structured output


# Step 1 - define the fields of the structured output as a Pydantic model

# NOTE: comment out one of the following 2 class definitions
# for a more detailed specification, go with the 2nd one


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


# class Classification(BaseModel):
#     sentiment: Literal["happy", "neutral", "sad"] = Field(
#         ..., description="The sentiment of the text"
#     )
#     aggressiveness: Literal[1, 2, 3, 4, 5] = Field(
#         ...,
#         description="Describes how aggressive the statement is; the higher the number, the more aggressive",
#     )
#     language: Literal["spanish", "english", "french", "german", "italian"] = Field(
#         ..., description="The language the text is written in"
#     )


# Step 2: bind our LLM to the structured output like so
structured_llm = llm.with_structured_output(Classification)

# Step 3: define our prompt for extraction of the structured output
prompt_template = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{user_input}
"""
)

# Step 4: Ask away
while True:
    console.print("[yellow]Your input: [/yellow]", end="")
    user_input = input().strip().lower()
    if len(user_input) <= 0:
        console.print("[red]Please enter some text![/red]")
        continue
    elif user_input in ["exit", "bye", "quit"]:
        console.print("[red]Quitting the application[/red]")
        break

    prompt = prompt_template.invoke({"user_input": user_input})
    console.print(f"[blue]Prompt:[/blue]\n{prompt}")

    # NOTE: I am asking the structured LLM, not LLM
    response = structured_llm.invoke(prompt)
    console.print(f"[green]AI: [/green]\n\n{response}")
