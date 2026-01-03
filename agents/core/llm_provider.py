"""
LLM Provider Interface

Abstracts away specific LLM implementations (Groq, OpenAI, Ollama, etc.)
Allows seamless provider switching while maintaining consistent API.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class LLMMessage:
    """A single message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    tool_calls: list[dict[str, Any]] = None  # List of {name, arguments}
    stop_reason: str = "end_turn"  # "end_turn", "tool_use", etc.


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: Conversation history
            tools: Available tools in OpenAI format
            temperature: Sampling temperature
            max_tokens: Max tokens in response
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with content and optional tool calls
        """
        pass


class GroqProvider(LLMProvider):
    """Groq API implementation of LLMProvider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq provider.

        Args:
            api_key: Groq API key (reads from GROQ_API_KEY env if not provided)
            model: Model name to use
        """
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Groq API."""
        # Convert our message format to Groq format
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        call_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

        if tools:
            call_kwargs["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]
            call_kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message

        # Extract tool calls if present
        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )


class OllamaProvider(LLMProvider):
    """Ollama implementation of LLMProvider for local models."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """
        Initialize Ollama provider.

        Args:
            base_url: Ollama server base URL
            model: Model name to use
        """
        from ollama import Client

        self.client = Client(host=base_url)
        self.model = model

    def generate(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Ollama."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        response = self.client.chat(
            model=self.model,
            messages=formatted_messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )

        # Ollama doesn't natively support tool calling, so we return raw response
        # Tool integration would require custom parsing or a wrapper
        return LLMResponse(
            content=response["message"]["content"],
            tool_calls=None,
            stop_reason="end_turn",
        )
