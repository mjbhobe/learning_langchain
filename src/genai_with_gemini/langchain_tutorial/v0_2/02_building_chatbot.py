"""
building_chatbot.py - building a chatbot with Anthropic Claude Sonnet/Groq (Llama)
"""

import os, sys
from dotenv import load_dotenv, find_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# load env variables from .env file
_ = load_dotenv()  # read local .env file


# create instance of the text model
sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
)

llama = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

from langchain_core.messages import HumanMessage

response = llama.invoke([HumanMessage(content="Hi! I'm Bob")])
print(response.content)
response = llama.invoke([HumanMessage(content="What's my name?")])
print(response.content)

print("-" * 80)

# let's add message history
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

# message store dict
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


model_with_msg_history = RunnableWithMessageHistory(llama, get_session_history)

# let's set a session ID for the session & call chat
config = {"configurable": {"session_id": "abc2"}}
response = model_with_msg_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
)
print(response.content)

# now ask a follow-up question
# since previous message is saved in memory, it should be able
# to understand that you are Bob
response = model_with_msg_history.invoke(
    [HumanMessage(content="Hey, what's my name?")],
    config=config,
)
print(response.content)

print("-" * 80)

# using prompt templates
