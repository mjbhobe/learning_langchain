# LangChain Foundation

This course introduces you to how you can build Agents using the LangChain framework. It progressively introduces you to all the Agent related features provided by LangChain:
* [Module1](Module1/readme.md) - covers the basic features such as creating a basic agent, adding system prompts, tools, memory and multi-modal inputs. It concludes with an interesting personal chef application that applies all the techniques learnt in this module.
* Module2 - we jump to more advanced concepts such as the Model Context protocol (MCP) << to do>>
* Module3 - << to do>>

Here are instructions on how to setup your local environment. We'll be using `uv` to manage my local environment and Python >=3.12.

## Installation
Download this repo from `git` using the following command:

```bash
$> git clone https://github.com/mjbhobe/learning_langchain.git
$> cd learning_langchain/src/langchain_foundation
```

Make a copy of `.env.example` file and add the API keys for your preferred LLM. We'll be using OpenAI as our LLM, but feel free to use your favourite.

```
# to use OpenAI-GPT as your LLM
OPENAI_API_KEY=<<add your OpenAI API key>>

# to use Google Gemini as your preferred LLM
GOOGLE_API_KEY=<< add your Google API key here >>

# to use Anthropoic's Claude models as your preferred LLMs
ANTHROPIC_API_KEY=<< add your Anthropic API key here >>

# we'll be using Tavily Search as a web-search tool.
# get your Tavily API key from https://app.tavily.com/home (under API Keys tab)
TAVILY_API_KEY=<< add your Tavily API key here >>

# we'll be using LangSmith to 'visually' test our agents

# get your LangSmith API key from Langsmith home page
LANGSMITH_API_KEY=<<add your LangSmith API key here>>
# set value of below key to the LangSmith project associated with above API key
LANGSMITH_PROJECT=<<add your LangSmith project name here>>
# set the below key to true
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
```

### Synch your local environment

```
$> uv sync
```
The above command will install all the needed packages for this project to your local `uv` enviroment.

That's it!

Now you can open this folder in your IDE (Visual Studio Code, Antigravity, Cursor, PyCharm - pick your favourite!) and run the notebooks as you do _after_ picking the local `uv` environment.




