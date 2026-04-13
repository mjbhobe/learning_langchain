# 01 Module 1

In this module, we'll introduce AI Agents in LangChain

1. We'll learn how to **initialize and interact with a chat model** using LangChain's `create_agent()` abstraction - see [foundation models](01_foundation_models.ipynb).
2. Then we'll explore **how to add system prompts** to our agent and how you can generate structured output from models - see [prompting and structured output](02_prompting_and_structured_output.ipynb).
3. Next, we'll then **integrate our agent with tools** - the ability to work with tools is what separates a regular chat model from an Agent.
4. Then we'll add **short term memory** to our agent, which will allow the agent to retain memory of previous messages, so you can actually have a back-and-forth conversation with it, which is the behaviour in #1.
5. Next we will **enable multi-modal inputs (images & audio) to our agent** - this will show you techniques of interfacing images & audio as inputs to our agent.
6. Finally, we'll apply all the techniques we learn't above and build a **Personal Chef Assistant** that can find recipies off the internet based on left-over ingredients in your fridge. Cool eh? 

We assume you have setup your local Python environment and API keys as described in the [readme.md](readme.md) file. If not, please do so before proceeding with the rest of this notebook.

Let's go!