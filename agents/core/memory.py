"""
Memory System

Manages conversation history and observations.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


@dataclass
class Observation:
    """A single observation from tool execution or reasoning."""

    timestamp: datetime = field(default_factory=datetime.now)
    observation_type: str = "tool"  # "tool", "reasoning", "error"
    content: str = ""
    tool_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.observation_type,
            "content": self.content,
            "tool_name": self.tool_name,
            "metadata": self.metadata,
        }


class Memory:
    """
    Manages short-term conversation history and long-term observations.

    Short-term: Conversation turns (messages for LLM context)
    Long-term: Observations from tool execution and reasoning
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize memory.

        Args:
            max_history: Maximum number of conversation turns to keep
        """
        self.max_history = max_history
        self.short_term: list[dict[str, str]] = []  # Conversation messages
        self.long_term: list[Observation] = []  # Observations log

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to short-term memory.

        Args:
            role: "system", "user", "assistant", or "tool"
            content: Message content
        """
        self.short_term.append({"role": role, "content": content})

        # Trim if exceeds max history
        if len(self.short_term) > self.max_history:
            self.short_term = self.short_term[-self.max_history :]

    def add_observation(
        self,
        observation_type: str,
        content: str,
        tool_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Add an observation to long-term memory.

        Args:
            observation_type: "tool", "reasoning", "error"
            content: Observation content
            tool_name: Name of tool if applicable
            metadata: Additional metadata
        """
        obs = Observation(
            observation_type=observation_type,
            content=content,
            tool_name=tool_name,
            metadata=metadata or {},
        )
        self.long_term.append(obs)

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get current conversation history for LLM context."""
        return self.short_term.copy()

    def get_observations(
        self, observation_type: Optional[str] = None, limit: Optional[int] = None
    ) -> list[Observation]:
        """
        Get observations, optionally filtered.

        Args:
            observation_type: Filter by type ("tool", "reasoning", "error")
            limit: Max number to return (most recent first)

        Returns:
            List of observations
        """
        obs = self.long_term.copy()

        if observation_type:
            obs = [o for o in obs if o.observation_type == observation_type]

        if limit:
            obs = obs[-limit:]

        return obs

    def clear_short_term(self) -> None:
        """Clear conversation history (keep observations)."""
        self.short_term = []

    def clear_all(self) -> None:
        """Clear all memory."""
        self.short_term = []
        self.long_term = []

    def to_dict(self) -> dict[str, Any]:
        """Export memory state to dictionary."""
        return {
            "short_term_messages": self.short_term,
            "long_term_observations": [obs.to_dict() for obs in self.long_term],
            "total_messages": len(self.short_term),
            "total_observations": len(self.long_term),
        }
