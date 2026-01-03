"""
Agent Loop Implementation

Core reasoning loop: thought → decision → action → observation → continue/finish
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from .llm_provider import LLMProvider, LLMMessage
from .tool import ToolRegistry
from .memory import Memory


@dataclass
class AgentConfig:
    """Configuration for the agent."""

    max_iterations: int = 10
    verbose: bool = True
    temperature: float = 0.7
    system_prompt: str = (
        "You are a helpful AI assistant. You have access to tools to help answer questions. "
        "When you need information, use the available tools. Always provide clear reasoning "
        "for your decisions. Respond with JSON containing your 'thought', 'action' (tool|respond), "
        "and if action is 'tool', provide 'tool_name' and 'tool_input'."
    )


@dataclass
class AgentState:
    """Current state of the agent."""

    iteration: int = 0
    status: str = "running"  # "running", "completed", "max_iterations", "error"
    final_response: Optional[str] = None
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    errors: list[str] = field(default_factory=list)


class Agent:
    """
    Core AI Agent with reasoning loop.

    Flow:
    1. Accept user goal/query
    2. Add to memory as user message
    3. Loop (up to max_iterations):
       a. Call LLM with conversation history
       b. Parse reasoning (thought, action, tool_name, tool_input)
       c. If action is 'tool': execute tool, add observation
       d. If action is 'respond': return response and break
    4. Return final state
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the agent.

        Args:
            llm_provider: LLM implementation to use
            tool_registry: Registry of available tools
            config: Agent configuration
        """
        self.llm = llm_provider
        self.tools = tool_registry
        self.config = config or AgentConfig()
        self.memory = Memory()
        self.state = AgentState()

    def run(self, user_query: str, verbose: Optional[bool] = None) -> AgentState:
        """
        Run the agent with a user query.

        Args:
            user_query: The user's goal or question
            verbose: Override config verbose setting

        Returns:
            AgentState with final response and trace
        """
        verbose = verbose if verbose is not None else self.config.verbose

        # Reset state for new run
        self.state = AgentState()
        self.memory.clear_short_term()

        # Add system prompt and user query
        self.memory.add_message("system", self.config.system_prompt)
        self.memory.add_message("user", user_query)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Agent Starting: {user_query}")
            print(f"{'='*60}\n")

        # Main agent loop
        while self.state.iteration < self.config.max_iterations:
            self.state.iteration += 1

            if verbose:
                print(f"Iteration {self.state.iteration}/{self.config.max_iterations}")

            # Get LLM response
            try:
                llm_response = self._get_llm_response(verbose)
            except Exception as e:
                error_msg = f"LLM error: {str(e)}"
                self.state.errors.append(error_msg)
                self.state.status = "error"
                if verbose:
                    print(f"ERROR: {error_msg}\n")
                break

            # Parse reasoning from response
            reasoning = self._parse_reasoning(llm_response.content)
            self.state.reasoning_trace.append(reasoning)

            if verbose:
                print(f"Thought: {reasoning.get('thought', 'N/A')}")
                print(f"Action: {reasoning.get('action', 'N/A')}")

            # Decide what to do
            action = reasoning.get("action", "respond").lower()

            if action == "tool":
                # Execute tool
                tool_name = reasoning.get("tool_name", "")
                tool_input = reasoning.get("tool_input", {})

                if verbose:
                    print(f"Tool: {tool_name}")
                    print(f"Input: {json.dumps(tool_input, indent=2)}")

                result = self.tools.execute(tool_name, **tool_input)
                self.state.tool_calls += 1

                # Add tool result to memory
                self.memory.add_message("assistant", llm_response.content)
                tool_result_msg = json.dumps(result)
                self.memory.add_message("tool", tool_result_msg)

                # Log observation
                self.memory.add_observation(
                    observation_type="tool",
                    content=tool_result_msg,
                    tool_name=tool_name,
                    metadata={"input": tool_input},
                )

                if verbose:
                    print(f"Result: {result}\n")

            elif action == "respond":
                # Extract final response
                self.state.final_response = reasoning.get(
                    "response", llm_response.content
                )
                self.state.status = "completed"
                self.memory.add_message("assistant", self.state.final_response)

                if verbose:
                    print(f"Response: {self.state.final_response}\n")
                break

            else:
                # Unknown action, treat as respond
                self.state.final_response = llm_response.content
                self.state.status = "completed"
                self.memory.add_message("assistant", llm_response.content)

                if verbose:
                    print(f"Response: {self.state.final_response}\n")
                break

        # Handle max iterations reached
        if self.state.iteration >= self.config.max_iterations and self.state.status == "running":
            self.state.status = "max_iterations"
            self.state.final_response = (
                "Max iterations reached. "
                + (self.state.final_response or "Unable to complete the task.")
            )
            if verbose:
                print(f"Max iterations ({self.config.max_iterations}) reached.\n")

        if verbose:
            print(f"{'='*60}")
            print(f"Final Status: {self.state.status}")
            print(f"Total Iterations: {self.state.iteration}")
            print(f"Tool Calls: {self.state.tool_calls}")
            print(f"Final Response: {self.state.final_response}")
            print(f"{'='*60}\n")

        return self.state

    def _get_llm_response(self, verbose: bool) -> Any:
        """Call the LLM with current conversation history."""
        messages = self.memory.get_conversation_history()
        llm_messages = [LLMMessage(role=m["role"], content=m["content"]) for m in messages]

        tools = self.tools.to_openai_format() if self.tools.get_all() else None

        return self.llm.generate(
            llm_messages,
            tools=tools,
            temperature=self.config.temperature,
            max_tokens=4096,
        )

    @staticmethod
    def _parse_reasoning(content: str) -> dict[str, Any]:
        """
        Parse reasoning from LLM response.

        Expected JSON format:
        {
            "thought": "...",
            "action": "tool" | "respond",
            "tool_name": "..." (if action is tool),
            "tool_input": {...} (if action is tool),
            "response": "..." (if action is respond)
        }
        """
        try:
            # Try to extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1

            if start != -1 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
            else:
                # Fallback: treat entire response as response
                return {
                    "thought": "No structured reasoning provided",
                    "action": "respond",
                    "response": content,
                }
        except json.JSONDecodeError:
            return {
                "thought": "Failed to parse reasoning",
                "action": "respond",
                "response": content,
            }

    def reset(self) -> None:
        """Reset agent state and memory for a new conversation."""
        self.state = AgentState()
        self.memory.clear_all()

    def get_memory_state(self) -> dict[str, Any]:
        """Get current memory state for inspection."""
        return self.memory.to_dict()
