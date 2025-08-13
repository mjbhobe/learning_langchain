""" 
hello_gemini_langchain.py: integrating Google Geimini into Langchain, a simple example

@author: Manish Bhobe
My experiments with Python, Data Science, Deep Learning & Generative AI.
Code is made available as-is & for learning purposes only, please use at your own risk!!
"""

import os
from dotenv import load_dotenv, find_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# load Google API key from .env file
_ = load_dotenv(find_dotenv())  # read local .env file

# create our model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"]

# and our prompt
tweet_prompt = PromptTemplate.from_template(
    """You are a content creator. Write me a funny tweet about {topic}"""
)
# and the chain
# tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt, verbose=True)
tweet_chain = tweet_prompt | llm

if __name__ == "__main__":
    topic = "Macbook Pro M3"
    # resp = tweet_chain.run(topic=topic)
    resp = tweet_chain.invoke({"topic": topic})
    print(resp.content)
