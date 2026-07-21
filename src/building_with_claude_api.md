# Building with Claude API

This course will teach developers how to integrate Claude AI into applications using the Anthropic API. We'll cover fundamental API operations, advanced prompting techniques, tool integration, and architectural patterns for building AI-powered systems. We will learn to implement conversational AI, retrieval-augmented generation, automated workflows, and leverage Claude's multimodal capabilities for processing text, images, and documents.

## What You'll Learn

* Set up and authenticate with the Anthropic API, including API key management and request configuration
* Implement single and multi-turn conversations with proper message formatting and context handling
* Configure system prompts and control model behavior using temperature, response streaming, and structured output formats
* Design and execute prompt evaluation workflows with test dataset generation and automated grading systems
* Apply prompt engineering techniques including XML tag structuring, example-based learning, and clear directive formulation
* Integrate Claude's tool use capabilities to extend functionality with custom tools, batch operations, and web search
* Build retrieval-augmented generation (RAG) systems with text chunking, embeddings, BM25 search, and contextual retrieval
* Utilize Claude's extended features including extended thinking mode, image analysis, PDF processing, and citation generation
* Implement prompt caching strategies to optimize API usage and reduce latency
* Develop Model Context Protocol (MCP) servers and clients for standardized tool and resource integration
* Deploy Anthropic Apps including Claude Code for automated development tasks and Computer Use for UI automation
* Architect agent-based systems with parallelization, chaining, and routing workflows.

## Overview of Claude Models

In this section we'll examine Claude's three model families and understand which one is best suited for your specific usecase. To help us understand how these models differ, we'll walk through each of the model's key characteristics and then look at a simple framework for picking the right one. 

|  | **Claude Opus** | **Claude Sonnet** | **Claude Haiku** |
| :--- | :--- | :--- | :--- |
| **Description** | Highest Level of Intelligence | Intelligent Model that balances quality, speed and cost | Most cost-effective and latency-optimized model |
| **Cost** | High | Medium | Low |
| **Comparative Latency** | Moderate | Fast | Fastest |
| **Supports Reasoning** | Yes | Yes | No |
| **Best Used For** | • Advanced software development, especially large-scale architecting<br>• Long running tasks that require sustained focus<br>• Strategic planning with multi-step problem solving<br>• Tasks that could benefit from advanced reasoning | • Common Coding tasks<br>• Document creation and editing<br>• Content marketing and copyrighting<br>• Data analysis and visualization<br>• Image analysis<br>• Process automation | • Quick code completions and suggestions<br>• Content moderation and filtering<br>• Data extraction and categorization<br>• Language translation<br>• Q&A Systems and knowledge retrieval<br>• Most high-volume and straightforward text |

You need to identify or figure out what matters most to your specific usecase:

* If intelligence is your top priority, meaning you have a complex task that really needs strong reasoning, then you'd probably want to use Opus. You are choosing quality over speed & cost.
* If speed is your priority, meaning you have real-time user-interactions, or you have some high volume processing, where you need to get responses back as fast as possible, then you'd want to choose Haiku.
* If you need more of a balance between intelligence, speed and cost, which if often the case for most of applications, then Sonnet is your best choice.

One important thing to note here is that many teams don't just pick one model and stick with it. Instead, you might use multiple different models in the same application. For example, Haiku for user-facing interactions where speed is really important, Sonnet for your main business logic, and Opus for really complex tasks that need some deeper reasoning. 

# Accessing Claude with the API

In this section we'll cover the full lifecycle of requests to the Anthropic API. We'll also take a brief look at what's going on behind the scenes inside of Claude. 

We'll begin with a standard straightforward chatbot app. Let's imagine you are building a web-app that wants to show a chat window to the user in a web-browser. When the user enters a message (prompt) and clicks send, they expect to see some response magically appear on the screen. We'll examine what goes on behind the scenes here to generate the final response to the user. 

We'll break this down into 5 separate steps:

<div align="center">
    <image src="images/claude_api_5_steps.png" alt="5 Steps of Claude API"/>
</div>

* **Request to Server**: when the user enters a prompt and clicks send, that request should be sent to some server that you, the developer, will implement. Don't access the Anthropic API directly from a web or mobile app. Whenever you call an Anthropic API, you are required to include a secret API key, and the best way to ensure that this API key stays secret is by never including it inside of your client side app and only making API calls from a server that you implement.

<div align="center">
    <image src="images/claude_api_request_to_server.png" alt="Step 1 - Request to Server"/>
</div>

* **Request to Anthropic API**: once your server received a user (client side) request, it will make make a request to Anthropic API. Usually you'll make this request using one of the SDKs that Anthropic has published - Anthropic has SDKs for TypeScript, Python, JavaScript, Go and Ruby. You don't have to use an SDK if you don't want to - you can make a plain HTTP request if you wish. 

    When you make  a request, you are required to pass on several pieces of data - in particular, you need to include an API Key, the name of the model (Opus, Sonnet, Haiku) you wish to run, 

<div align="center">
    <image src="images/claude_api_request_to_api.png" alt="Step 1 - Request to Anthropic API"/>
</div>
