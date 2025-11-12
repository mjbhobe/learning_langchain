# RAG from Scratch
RAG of Retrieval Augmented Generation is one of the most popular applications in Generative AI. 

The motivation behind RAG is that most of the world's data is private, whereas LLMs are trained on publicly available data. So they can easily answer question on public data, such as "Tell me about Sachin Tendulkar". However they have no knowledge about your private/confidential data, such as your business proposals to customer X. So an LLM will struggle to answer a question such as "what type of staffing model did we propose to customer X for their Insurance UW proposal?".

Also LLM's have a _training cutoff_, meaning they are trained with data upto a certain data, such as Sept 2023. It will therefore struggle to answer questions on more recent events too - such as "Who won the ICC T20 mens trophy in 2025?"

What's really interesting with modern LLMs, such as Google Gemini Flash or Anthropic Claude or OpenAI GPT is the size of the context window is getting larger and larger (~1 Mn tokens, which is approximately 1000 pages of text!). This is making it increasingly feasible to feed external private data to an LLM - something it has never seen; could be your own corporate data for example - and the LLM can answer question off that data. This is the main motivation around RAG.

You can think of RAG as 3 very general steps as shown in the figure below:

![RAG Pipeline](images/rag_pipeline.png)

1. There is the process of _Indexing_ of external data
