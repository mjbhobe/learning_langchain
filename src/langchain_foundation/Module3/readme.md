# LangChain Foundation - Module 3

This is Module 3. In this module, you'll apply whatever you've learnt so far in [Module1](../Module1/readme.md) and [Module2](../Module2/readme.md) and push your agents beyond the prototype stage.

Here's what we'll cover in this module:

1. We'll start by by introducing **middleware**. One of the most important important concepts in modern Agent design. Middleware lets you intercept and customize your agents execution at every step. We'll learn how to dynamically swap tools, adjust prompts on the fly and even change the underlying model based on the situation your Agent finds itself in. See [01_middleware.ipynb](01_middleware.ipynb).
2. Next we'll discuss **long conversation management**. Context windows are finite, but great conversations shouldn't be! In this module, we'll learn how to summarize, compress and intelligently retain information so that your agent can stay coherent over hours or days of interaction, without forgetting what matters.
3. Next we'll look at **human in the loop** patterns. Not every action should be left entirely to an AI system, especially sensitive ones. We'll learn how to insert improval checkpoints where our agent and it's human counterpart work together seamlessly. 
4. Up until now, our Agents have been powerful, but they have been fairly _static_. Fixed tools, fixed prompts, fixed models. Real-world AI applications don't work like that
5. Finally, we'll close this module by building an Email Assistant capable of automating our entire inbox _safely_, _reliably_, and with all the custom logic we have learnt above.
