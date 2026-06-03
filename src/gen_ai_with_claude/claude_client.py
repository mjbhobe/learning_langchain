"""default claude client (convenience class)"""

import os
from typing import Optional
from dotenv import load_dotenv

import anthropic
from anthropic.types import Message


class ClaudeClient:
    """Convenience wrapper around the Anthropic client for single-turn interactions.

    Attributes:
        DEFAULT_CLAUDE_MODEL: Fallback model used when none is specified at construction.
        DEFAULT_MAX_TOKENS: Fallback token limit used when none is specified at construction.
        client: The underlying ``anthropic.Client`` instance.
        model: The Claude model identifier used for all requests.
        max_tokens: The maximum number of tokens allowed in each response.
    """

    DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Message:
        """Initializes the ClaudeClient with a model and token limit.

        Loads environment variables from a .env file and creates an Anthropic
        client instance using credentials from the environment.

        Args:
            model: The Claude model identifier to use for requests. Defaults to
                ``DEFAULT_CLAUDE_MODEL`` if not provided.
            max_tokens: The maximum number of tokens to generate in a response.
                Defaults to ``DEFAULT_MAX_TOKENS`` if not provided.
        """
        load_dotenv(override=True)
        self.client = anthropic.Client()
        self.model = model or self.DEFAULT_CLAUDE_MODEL
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

    def ask(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Sends a prompt to Claude and returns the raw API response.

        Constructs a messages request with the given user prompt and optional
        system prompt, then calls the Anthropic messages API.

        Args:
            prompt: The user message to send to Claude.
            system_prompt: An optional system-level instruction prepended to the
                conversation to guide Claude's behavior. If ``None``, no system
                message is added.

        Returns:
            The ``Message`` object returned by the Anthropic API, containing
            the model's response and metadata.
        """
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        if system_prompt:
            kwargs["messages"].insert(0, {"role": "system", "content": system_prompt})

        response = self.client.messages.create(**kwargs)
        return response


# test it
if __name__ == "__main__":
    client = ClaudeClient()
    response = client.ask(
        "What is the difference between __str__ and __repr__ in Python?"
    )
    print(response)
    # print(response.content[0].text)
