"""
Quick Reference: AI Agent Framework

A cheat sheet for using the framework components.
"""

# ============================================================================
# SETUP
# ============================================================================

# Installation
# pip install -r requirements.txt

# Environment
# export GROQ_API_KEY="your_key_here"
# Or create .env file with GROQ_API_KEY=...

# ============================================================================
# QUICK START: 5 LINES
# ============================================================================

from core import Agent, AgentConfig, GroqProvider, ToolRegistry

llm = GroqProvider()
tools = ToolRegistry()
agent = Agent(llm, tools, AgentConfig())
state = agent.run("What is 5 + 5?")
print(state.final_response)

# ============================================================================
# LLM PROVIDERS
# ============================================================================

from core import GroqProvider, OllamaProvider, LLMProvider

# Groq
llm = GroqProvider(
    api_key="your_key",  # Optional, reads from GROQ_API_KEY env
    model="llama-3.3-70b-versatile"
)

# Ollama (local)
llm = OllamaProvider(
    base_url="http://localhost:11434",
    model="llama2"
)

# Custom provider
class MyLLM(LLMProvider):
    def generate(self, messages, tools=None, temperature=0.7, max_tokens=4096, **kwargs):
        # Your implementation
        from core import LLMResponse
        return LLMResponse(
            content="Response text",
            tool_calls=[{"name": "tool", "arguments": "{}"}]
        )

# ============================================================================
# TOOLS
# ============================================================================

from core import ToolRegistry, Tool, FunctionTool
from typing import Any

# Option 1: Register a function
def my_calculator(expression: str) -> str:
    """Calculate math expressions"""
    return str(eval(expression))

tools = ToolRegistry()
tools.register_function(
    my_calculator,
    name="calc",
    description="Evaluate math expressions"
)

# Option 2: Custom tool class
class WebSearchTool(Tool):
    @property
    def name(self) -> str:
        return "search"
    
    @property
    def description(self) -> str:
        return "Search the web"
    
    def execute(self, **kwargs) -> dict[str, Any]:
        query = kwargs.get("query", "")
        # Your implementation
        return {"result": f"Search results for {query}"}
    
    def _build_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }

tools.register(WebSearchTool())

# Inspect tools
all_tools = tools.get_all()  # dict of name -> Tool
tool = tools.get("calc")     # Get specific tool
openai_format = tools.to_openai_format()  # For API calls

# Execute a tool
result = tools.execute("calc", expression="2+2")
# {'result': '4'}

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

from core import AgentConfig

config = AgentConfig(
    max_iterations=10,      # Max reasoning steps
    verbose=True,           # Print each step
    temperature=0.7,        # LLM randomness (0-1)
    system_prompt="You are a helpful assistant..."  # System message
)

# ============================================================================
# RUNNING AGENTS
# ============================================================================

from core import Agent

agent = Agent(llm, tools, config)

# Single run
state = agent.run("Your question here", verbose=True)

# Access results
print(state.final_response)      # Final answer
print(state.status)              # "completed", "max_iterations", "error"
print(state.iteration)           # Number of iterations
print(state.tool_calls)          # Number of tool uses
print(state.reasoning_trace)     # List of reasoning steps
print(state.errors)              # Any errors encountered

# Reset for new conversation
agent.reset()
state2 = agent.run("New question")

# ============================================================================
# MEMORY ACCESS
# ============================================================================

# Conversation history
history = agent.memory.get_conversation_history()
# [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]

# Add message manually
agent.memory.add_message("user", "New message")

# Observations (tool executions)
observations = agent.memory.get_observations()  # All observations
tool_obs = agent.memory.get_observations(observation_type="tool")
last_5 = agent.memory.get_observations(limit=5)

# Full memory state
memory_dict = agent.get_memory_state()
# {
#   "short_term_messages": [...],
#   "long_term_observations": [...],
#   "total_messages": 10,
#   "total_observations": 5
# }

# Add custom observation
agent.memory.add_observation(
    observation_type="reasoning",
    content="My reasoning about something",
    metadata={"key": "value"}
)

# Clear memory
agent.memory.clear_short_term()  # Just conversation
agent.memory.clear_all()         # Everything

# ============================================================================
# REASONING TRACE
# ============================================================================

state = agent.run("Query")

for i, step in enumerate(state.reasoning_trace):
    print(f"Step {i}:")
    print(f"  Thought: {step.get('thought')}")
    print(f"  Action: {step.get('action')}")       # "tool" or "respond"
    print(f"  Tool: {step.get('tool_name')}")      # If tool used
    print(f"  Input: {step.get('tool_input')}")    # Tool parameters
    print(f"  Response: {step.get('response')}")   # If responding

# ============================================================================
# COMPLETE EXAMPLE
# ============================================================================

from core import Agent, AgentConfig, GroqProvider, ToolRegistry
import math

# Define tools
def calculate(expr: str) -> str:
    """Evaluate math"""
    return str(eval(expr))

def get_info(topic: str) -> str:
    """Get information"""
    info = {
        "python": "Python is a programming language",
        "ai": "AI is artificial intelligence"
    }
    return info.get(topic, "Unknown topic")

# Setup
llm = GroqProvider(model="llama-3.3-70b-versatile")
tools = ToolRegistry()
tools.register_function(calculate, name="calculate", description="Math")
tools.register_function(get_info, name="info", description="Get info")

config = AgentConfig(
    max_iterations=5,
    verbose=True,
    system_prompt="Help the user. Use tools when needed."
)

# Run
agent = Agent(llm, tools, config)
state = agent.run("What is 2+2 and tell me about Python?")

print(f"\nAnswer: {state.final_response}")
print(f"Status: {state.status}")
print(f"Iterations: {state.iteration}")
print(f"Tool calls: {state.tool_calls}")

# ============================================================================
# COMMON PATTERNS
# ============================================================================

# Pattern 1: Multiple sequential queries
agent = Agent(llm, tools)
for query in ["Query 1", "Query 2", "Query 3"]:
    state = agent.run(query)
    print(f"{query}: {state.final_response}")
    agent.reset()  # Fresh memory for next query

# Pattern 2: Inspect reasoning
state = agent.run(query)
for step in state.reasoning_trace:
    if step['action'] == 'tool':
        print(f"Used {step['tool_name']} with {step['tool_input']}")

# Pattern 3: Conditional tool registration
tools = ToolRegistry()
if use_calculator:
    tools.register_function(calculate, "calc", "Calc")
if use_search:
    tools.register_function(search, "search", "Search")

# Pattern 4: Reuse agent with different configs
agent = Agent(llm, tools)
fast_state = agent.run(query, verbose=False)  # Quick
slow_state = agent.run(query, verbose=True)   # Detailed

# Pattern 5: Error handling
try:
    state = agent.run(query)
    if state.status == "error":
        print(f"Errors: {state.errors}")
except Exception as e:
    print(f"Failed: {e}")

# ============================================================================
# DEBUGGING
# ============================================================================

# Enable verbose mode to see all steps
state = agent.run(query, verbose=True)

# Inspect memory after run
print(agent.get_memory_state())

# Check reasoning for each iteration
for i, reasoning in enumerate(state.reasoning_trace):
    print(f"Iteration {i}: {reasoning}")

# Track tool execution
for obs in agent.memory.get_observations(observation_type="tool"):
    print(f"Tool: {obs.tool_name}")
    print(f"Result: {obs.content}")

# Check errors
if state.errors:
    for error in state.errors:
        print(f"ERROR: {error}")

# ============================================================================
# AGENT STATE REFERENCE
# ============================================================================

"""
state.status: str
  - "completed"      ✓ Task finished successfully
  - "max_iterations" ! Reached iteration limit
  - "running"        ~ Still in progress (shouldn't see this after run())
  - "error"          ✗ An error occurred

state.iteration: int
  Number of reasoning steps executed

state.final_response: str
  The final answer from the agent

state.tool_calls: int
  Number of times a tool was executed

state.reasoning_trace: list[dict]
  Each step's reasoning with thought, action, tool_name, etc.

state.errors: list[str]
  Any errors that occurred during execution
"""

# ============================================================================
# IMPORTS CHEAT SHEET
# ============================================================================

"""
Core Components:
  from core import Agent, AgentConfig, AgentState
  from core import LLMProvider, GroqProvider, OllamaProvider
  from core import Tool, ToolRegistry, FunctionTool
  from core import Memory, Observation

Data Types:
  from core import LLMMessage, LLMResponse

Or import everything:
  from core import *
"""
