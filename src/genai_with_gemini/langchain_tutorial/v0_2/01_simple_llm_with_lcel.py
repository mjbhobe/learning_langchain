"""
simple_llm_with_lcel.py - calling LLM with simple message
NOTE: we'll be using Anthropic's Claude Sonnet for this example
"""

import os, sys
from dotenv import load_dotenv, find_dotenv

from langchain_anthropic import ChatAnthropic

# load env variables from .env file
_ = load_dotenv()  # read local .env file


# create instance of the text model
model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
)

# from langchain_core.messages import HumanMessage, SystemMessage

# messages = [
#     SystemMessage(content="Translate the following from English into Italian"),
#     HumanMessage(content="Good morning! Have a fantastic day."),
# ]

# response = model.invoke(messages)
# print(f"Direct from LLM: {response.content}")

# # let's use an OutputParser
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
# print(f"From parser: {parser.invoke(response)}")


from langchain_core.prompts import ChatPromptTemplate

system_message = "Translate the following from {source_language} to {target_language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_message), ("user", "{user_message}")]
)
src_lang = "English"
tgt_lang = "French"
msg = "Langchain is a great framework to develop LLM applications"
print(f'Translating "{msg}" from {src_lang} to {tgt_lang}')
result = prompt_template.invoke(
    {"source_language": src_lang, "target_language": tgt_lang, "user_message": msg}
)
print(f"Prompt template: {result}")
chain = prompt_template | model | parser
result = chain.invoke(
    {"source_language": src_lang, "target_language": tgt_lang, "user_message": msg}
)
print(result)
