"""
Tool System

Defines the Tool base class and ToolRegistry for pluggable tool management.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, get_type_hints
import json
import inspect


class Tool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the tool."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the tool.

        Args:
            **kwargs: Parameters from the agent

        Returns:
            dict with execution result
        """
        pass

    def to_openai_format(self) -> dict[str, Any]:
        """Convert tool to OpenAI function calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._build_parameters(),
        }

    @abstractmethod
    def _build_parameters(self) -> dict[str, Any]:
        """Build OpenAI-format parameters schema."""
        pass


class FunctionTool(Tool):
    """Adapter to wrap a Python function as a Tool."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Create a FunctionTool from a callable.

        Args:
            func: The function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
        """
        self._func = func
        self._name = name or func.__name__
        self._description = description or (func.__doc__ or "No description provided")

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the wrapped function."""
        try:
            result = self._func(**kwargs)
            return {"result": result, "error": None}
        except Exception as e:
            return {"result": None, "error": str(e)}

    def _build_parameters(self) -> dict[str, Any]:
        """Extract parameters from function signature."""
        sig = inspect.signature(self._func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            prop = {"type": "string", "description": ""}

            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                type_str = str(param.annotation)
                if "int" in type_str:
                    prop["type"] = "integer"
                elif "float" in type_str:
                    prop["type"] = "number"
                elif "bool" in type_str:
                    prop["type"] = "boolean"

            # Check if parameter has a default (not required if it does)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

            properties[param_name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


class ToolRegistry:
    """Manages and provides access to available tools."""

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Register a function as a tool.

        Args:
            func: Function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
        """
        tool = FunctionTool(func, name, description)
        self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> dict[str, Tool]:
        """Get all registered tools."""
        return self._tools.copy()

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI function calling format."""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool

        Returns:
            dict with execution result
        """
        tool = self.get(tool_name)
        if not tool:
            return {"result": None, "error": f"Tool '{tool_name}' not found"}

        return tool.execute(**kwargs)
