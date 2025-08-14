""" hello_gemini.py: Hello Google Gemini
This module introduces Google Gemini API and tests whether everything is 
setup correctly to use the Google Gemini LLM

@author: Manish Bhobe
My experiments with Python, Data Science, Deep Learning & Generative AI.
Code is made available as-is & for learning purposes only, please use at your own risk!!
"""

import os
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

import google.generativeai as genai

# load env variables from .env file
_ = load_dotenv(override=True)  # read local .env file
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# create an instance of Gemini LLM
# alternatively, use "gemini-pro" instead of "gemini-2.5-flash"
model = genai.GenerativeModel("gemini-2.5-flash")

console = Console()
prompt: str = "What is Google Gemini?"
console.print(f"[green]{prompt}[/green]")
response = model.generate_content(prompt)
# response = model.invoke(prompt)
markdown = Markdown(response.text)
console.print(markdown)
# print(remove_markdown(response.text))
