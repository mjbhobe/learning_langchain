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


# create instance of the text model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

from langchain_core.messages import HumanMessage, SystemMessage

s: str = "Calls without using Prompt Templates"
print(s + "-" * (80 - len(s)))
messages = [
    SystemMessage("Translate the following from English to Spanish"),
    HumanMessage("Google Gemini is one of the most capable multi-modal LLMs"),
]

response = model.invoke(messages)
print(response)
print(response.content)
print("-" * 80)

# create a prompt template
s1: str = "Calls using Prompt Templates"
print(s1 + "-" * (80 - len(s)))
from langchain_core.prompts import ChatPromptTemplate

# "language" and "user_message" are 'fill-in' variables
system_message: str = "Translate the following sentence from English to {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_message), ("user", "{user_message}")]
)

# let's ask user for values of fill-in variables
to_lang: str = input("Enter target translation language: ")
sentence: str = input(f"Enter text you want translated from English to {to_lang}: ")
# fill-in the values into template
prompt = prompt_template.invoke({"language": to_lang, "user_message": sentence})
print(f"Prompt: {prompt.to_messages()}")
response = model.invoke(prompt)
print(response.content)
print("-" * 80)
