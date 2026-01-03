"""
Framework Example: Multi-tool Agent

Demonstrates the complete agent framework with:
- LLM provider (Groq)
- Multiple tools (calculator, web search mock, fact retriever)
- Memory management
- Agent reasoning loop
"""

from core import (
    Agent,
    AgentConfig,
    GroqProvider,
    ToolRegistry,
)
import json
import math


# ============================================================================
# Tool Definitions
# ============================================================================

def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression (e.g., "2 + 2 * 3")

    Returns:
        Result as string
    """
    try:
        # Safe evaluation (limited scope)
        safe_dict = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "log": math.log,
            "pi": math.pi,
            "e": math.e,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def search_web(query: str) -> str:
    """
    Mock web search tool (in real scenario, would call actual search API).

    Args:
        query: Search query

    Returns:
        Mock search results as string
    """
    # Mock results database
    mock_results = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "machine learning": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "ai agents": "AI agents are autonomous systems that perceive their environment and take actions to achieve goals.",
        "llm": "Large Language Models (LLMs) are neural networks trained on vast amounts of text data.",
    }

    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value

    return "No relevant results found for your query."


def get_fact(topic: str) -> str:
    """
    Retrieve a fact about a topic.

    Args:
        topic: Topic to get a fact about

    Returns:
        A fact as a string
    """
    facts = {
        "python": "Python was created by Guido van Rossum and first released in 1991.",
        "earth": "Earth is the third planet from the Sun and has one natural satellite, the Moon.",
        "ai": "The term 'Artificial Intelligence' was coined by John McCarthy in 1956.",
        "space": "The observable universe is estimated to be about 93 billion light-years in diameter.",
    }

    for key, value in facts.items():
        if key.lower() in topic.lower():
            return value

    return f"No fact found for topic: {topic}"


# ============================================================================
# Main Agent Setup
# ============================================================================

def main():
    """Run the framework example."""

    print("\n" + "="*70)
    print("AI AGENT FRAMEWORK EXAMPLE")
    print("="*70 + "\n")

    # 1. Initialize LLM provider
    print("📡 Initializing LLM Provider (Groq)...")
    llm = GroqProvider(model="llama-3.3-70b-versatile")

    # 2. Initialize tool registry and register tools
    print("🔧 Registering Tools...")
    tools = ToolRegistry()
    tools.register_function(
        calculate,
        name="calculate",
        description="Evaluate mathematical expressions. Supports +, -, *, /, sqrt, sin, cos, pi, e, etc.",
    )
    tools.register_function(
        search_web,
        name="search_web",
        description="Search the web for information about a topic.",
    )
    tools.register_function(
        get_fact,
        name="get_fact",
        description="Retrieve an interesting fact about a given topic.",
    )

    print(f"✓ Registered tools: {list(tools.get_all().keys())}\n")

    # 3. Configure agent
    config = AgentConfig(
        max_iterations=10,
        verbose=True,
        temperature=0.7,
        system_prompt=(
            "You are a helpful AI assistant with access to tools for calculations, web search, and facts. "
            "When answering questions, think step-by-step. Use tools when appropriate. "
            "Always respond with JSON in this format:\n"
            '{\n'
            '  "thought": "Your reasoning here",\n'
            '  "action": "tool" or "respond",\n'
            '  "tool_name": "tool name if action is tool",\n'
            '  "tool_input": {"param": "value"},\n'
            '  "response": "Your final answer if action is respond"\n'
            "}\n"
        ),
    )

    # 4. Create and run agent
    print("🤖 Creating Agent...\n")
    agent = Agent(llm, tools, config)

    # Example queries
    queries = [
        "What is 25 * 4 + 10? And what is the square root of 144?",
        "Tell me a fact about Python and what 2^10 equals.",
    ]

    for query in queries:
        print(f"\n{'='*70}")
        print(f"USER QUERY: {query}")
        print(f"{'='*70}\n")

        state = agent.run(query)

        # Display results
        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        print(f"Status: {state.status}")
        print(f"Iterations: {state.iteration}")
        print(f"Tool Calls: {state.tool_calls}")
        print(f"\nFinal Response:\n{state.final_response}")

        # Show reasoning trace
        if state.reasoning_trace:
            print("\n" + "-"*70)
            print("REASONING TRACE")
            print("-"*70)
            for i, step in enumerate(state.reasoning_trace, 1):
                print(f"\nStep {i}:")
                print(f"  Thought: {step.get('thought', 'N/A')}")
                print(f"  Action: {step.get('action', 'N/A')}")
                if step.get("tool_name"):
                    print(f"  Tool: {step.get('tool_name')}")

        # Show memory state
        print("\n" + "-"*70)
        print("MEMORY STATE")
        print("-"*70)
        memory = agent.get_memory_state()
        print(f"Messages: {memory['total_messages']}")
        print(f"Observations: {memory['total_observations']}")

        print("\n")


if __name__ == "__main__":
    main()
