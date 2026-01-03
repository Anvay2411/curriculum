# AI Agent Framework

A modular, extensible framework for building intelligent agents with tool calling, reasoning loops, and memory management. Includes both learning examples and production-ready core components.

## 🎯 Overview

This framework provides:
- **LLM-Agnostic Provider Interface**: Support for Groq, OpenAI, Ollama, etc.
- **Tool System**: Pluggable tools with automatic OpenAI function calling format conversion
- **Memory Management**: Short-term conversation history + long-term observation logging
- **Agent Loop**: Structured reasoning (thought → plan → act → observe)
- **Learning Examples**: Simple calculator agent and structured response examples

## 📋 Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux
- Internet connection for API calls

## 📁 Project Structure

```
agents/
├── core/                          # Framework Core
│   ├── __init__.py               # Framework exports
│   ├── llm_provider.py           # LLM abstraction (Groq, OpenAI, Ollama)
│   ├── tool.py                   # Tool base class & registry
│   ├── memory.py                 # Short-term + long-term memory
│   └── agent.py                  # Main agent loop
├── simple_agent.py               # Learning example: Simple calculator
├── agent_structure.py            # Learning example: Structured responses
├── framework_example.py          # Framework demo: Multi-tool agent
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🏗️ Architecture

### Core Components

#### 1. LLM Provider (`core/llm_provider.py`)
Abstracts LLM implementations with a common interface:
- `LLMProvider`: Abstract base class
- `GroqProvider`: Groq API implementation
- `OllamaProvider`: Local Ollama models
- Easily extensible for OpenAI, Claude, etc.

#### 2. Tool System (`core/tool.py`)
Pluggable tool architecture:
- `Tool`: Base class for all tools
- `FunctionTool`: Adapter to wrap Python functions
- `ToolRegistry`: Manages available tools
- Automatic conversion to OpenAI function calling format

#### 3. Memory (`core/memory.py`)
Structured memory management:
- **Short-term**: Conversation history for LLM context
- **Long-term**: Observation log of all tool executions and reasoning
- Methods for querying, filtering, and exporting

#### 4. Agent Loop (`core/agent.py`)
Main reasoning loop:
1. Accept user query
2. Call LLM with conversation history
3. Parse structured reasoning (JSON)
4. Decide: use tool or respond
5. Execute tool or return final answer
6. Repeat until max iterations or completion

## 🚀 Quick Start

### Step 1: Install uv (Python Package Manager)

uv is a fast Python package manager that we'll use to manage our environment and dependencies.

#### Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Alternative (pip):
```bash
pip install uv
```

### Step 2: Create and Activate Virtual Environment

```bash
# Create a new virtual environment
uv venv

# Activate the environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (CMD):
.venv\Scripts\activate.bat

# macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all dependencies from requirements.txt
uv pip install -r requirements.txt

# Or install individually:
uv add groq instructor pydantic python-dotenv

# Or using pip if you prefer:
pip install -r requirements.txt
```

### Step 4: Set Up Groq API Key

1. **Get your API key**:
   - Visit [Groq Console](https://console.groq.com/)
   - Sign up for a free account
   - Navigate to API Keys section
   - Create a new API key

2. **Set the environment variable**:

   #### Windows (PowerShell):
   ```powershell
   $env:GROQ_API_KEY="your_api_key_here"
   ```

   #### Windows (CMD):
   ```cmd
   set GROQ_API_KEY=your_api_key_here
   ```

   #### macOS/Linux:
   ```bash
   export GROQ_API_KEY="your_api_key_here"
   ```

   #### Alternative: Create a .env file
   ```bash
   # Create .env file in project root
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```

## 🎯 Running the Learning Scripts

### Simple Agent (Calculator)

The `simple_agent.py` demonstrates basic function calling with a calculator tool:

```bash
python simple_agent.py
```

**What it does:**
- Uses Groq's Llama 3.3 70B model
- Implements function calling for mathematical calculations
- Shows conversation flow with tool usage
- Demonstrates error handling

**Example interaction:**
```
Input: "What is 25 * 4 + 10?"
Output: "The result of 25 * 4 + 10 is 110."
```

### Agent Structure (Structured Response)

The `agent_structure.py` demonstrates structured agent responses using Pydantic models and the instructor library:

```bash
python agent_structure.py
```

**What it does:**
- Uses instructor library for structured outputs
- Implements Pydantic models for type safety
- Demonstrates tool schema definition
- Shows structured tool call parsing

**Example interaction:**
```
Input: "What's the weather like in San Francisco?"
Output: 
Input: What's the weather like in San Francisco?
Tool: get_weather_info
Parameters: {"location": "San Francisco"}
```

#### Additional Dependencies for Agent Structure

This script requires additional packages:

```bash
# Install additional dependencies
uv add instructor pydantic

# Or using pip:
pip install instructor pydantic
```

## 🚀 Using the Framework

### Basic Example

```python
from core import Agent, AgentConfig, GroqProvider, ToolRegistry

# 1. Initialize LLM
llm = GroqProvider(model="llama-3.3-70b-versatile")

# 2. Create tool registry and add tools
tools = ToolRegistry()
tools.register_function(
    my_function,
    name="my_tool",
    description="What this tool does"
)

# 3. Configure agent
config = AgentConfig(
    max_iterations=10,
    verbose=True,
    system_prompt="You are a helpful assistant..."
)

# 4. Create and run agent
agent = Agent(llm, tools, config)
state = agent.run("User query here")

# 5. Access results
print(state.final_response)
print(f"Status: {state.status}")
print(f"Tool calls: {state.tool_calls}")
```

### Running the Framework Example

```bash
python framework_example.py
```

This demonstrates:
- Multiple tools (calculator, web search, fact retriever)
- Complex reasoning flow
- Memory management
- Verbose iteration tracking

### Complete Example: Custom Agent

See `framework_example.py` for a complete implementation with:
- 3 different tools
- Complex reasoning
- Error handling
- Memory inspection

## 🛠️ Extending the Framework

### Adding a New Tool

```python
from core import Tool
from typing import Any

class CustomTool(Tool):
    @property
    def name(self) -> str:
        return "my_custom_tool"
    
    @property
    def description(self) -> str:
        return "Description of what the tool does"
    
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        # Your implementation
        result = do_something(kwargs.get("param"))
        return {"result": result}
    
    def _build_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param"]
        }

# Register it
tools = ToolRegistry()
tools.register(CustomTool())
```

### Using Different LLM Providers

```python
# Groq (default)
llm = GroqProvider(model="llama-3.3-70b-versatile")

# Ollama (local)
llm = OllamaProvider(base_url="http://localhost:11434", model="llama2")

# Your custom provider
class MyProvider(LLMProvider):
    def generate(self, messages, tools=None, ...):
        # Your implementation
        pass
```

### Customizing Agent Behavior

```python
config = AgentConfig(
    max_iterations=15,          # Increase iteration limit
    verbose=False,              # Disable verbose output
    temperature=0.3,            # More deterministic
    system_prompt="Custom prompt here..."
)

agent = Agent(llm, tools, config)
```

### Accessing Agent Memory

```python
state = agent.run(query)

# Get full memory state
memory_dict = agent.get_memory_state()
print(memory_dict["total_messages"])
print(memory_dict["total_observations"])

# Short-term conversation history
conversation = agent.memory.get_conversation_history()

# Long-term observations
observations = agent.memory.get_observations(
    observation_type="tool",  # Optional filter
    limit=5  # Get last 5
)
```

## 📊 Agent State and Reasoning Trace

After running an agent, inspect detailed execution:

```python
state = agent.run(query)

# Overall state
print(f"Status: {state.status}")  # "completed", "max_iterations", "error"
print(f"Iterations: {state.iteration}")
print(f"Tool calls: {state.tool_calls}")
print(f"Final response: {state.final_response}")

# Reasoning trace (JSON from each iteration)
for i, reasoning in enumerate(state.reasoning_trace):
    print(f"Step {i}: {reasoning['thought']}")
    print(f"  Action: {reasoning['action']}")
    if reasoning.get('tool_name'):
        print(f"  Tool: {reasoning['tool_name']}")
```

## 🔄 Running on WSL (Windows Subsystem for Linux)

If you prefer to run the scripts on WSL for a Linux environment:

1. **Open WSL Terminal**:
```bash
# From Windows Command Prompt or PowerShell
wsl
```

2. **Navigate to your project directory**:
```bash
# If your project is on Windows drive C:
cd /mnt/c/Users/akulshre/OneDrive\ -\ Intel\ Corporation/Desktop/Development/Agents/

# Or copy the project to WSL home directory
cp -r /mnt/c/Users/akulshre/OneDrive\ -\ Intel\ Corporation/Desktop/Development/Agents ~/agents
cd ~/agents
```

3. **Set up Python environment in WSL**:
```bash
# Install uv in WSL (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv add groq instructor pydantic
```

4. **Set environment variable in WSL**:
```bash
export GROQ_API_KEY="your_api_key_here"

# Or add to ~/.bashrc for persistence
echo 'export GROQ_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

5. **Run the scripts**:
```bash
python simple_agent.py
python agent_structure.py
```

## 🧠 Understanding the Code

### Simple Agent Components

1. **Groq Client**: Direct connection to Groq API for LLM inference
2. **Function Definition**: `calculate()` function that evaluates math expressions
3. **Tool Schema**: JSON schema defining available functions for the LLM
4. **Conversation Flow**: Multi-turn conversation with function calling
5. **Error Handling**: Graceful handling of invalid expressions

### Agent Structure Components

1. **Instructor Library**: Enables structured outputs from LLMs using Pydantic models
2. **Pydantic Models**: Type-safe data structures for tool calls and responses
3. **BaseModel Classes**: `ToolCall` and `ResponseModel` for structured parsing
4. **JSON Mode**: Forces the LLM to respond in structured JSON format
5. **Tool Schema**: Defines available tools and their parameter requirements

### Comparison: Simple vs Structured Approach

| Feature | Simple Agent | Agent Structure |
|---------|-------------|-----------------|
| **Output Format** | Free text | Structured JSON |
| **Type Safety** | None | Pydantic validation |
| **Tool Parsing** | Manual JSON parsing | Automatic model parsing |
| **Error Handling** | Basic try/catch | Pydantic validation |
| **Complexity** | Lower | Higher |
| **Use Case** | Simple interactions | Production systems |

### Function Calling Flow

1. **Initial Request**: User sends a mathematical query
2. **LLM Analysis**: Model determines if a function call is needed
3. **Function Execution**: Calculator function processes the expression
4. **Result Integration**: Function result is added to conversation
5. **Final Response**: LLM generates human-readable response

## 🔧 Customization

### Adding New Functions

To extend the agent with new capabilities:

1. **Define your function**:
```python
def get_weather(city):
    """Get weather information for a city"""
    # Your implementation here
    return json.dumps({"weather": "sunny", "temperature": "72°F"})
```

2. **Add to tools schema**:
```python
tools = [
    # Existing calculate tool...
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name",
                    }
                },
                "required": ["city"],
            },
        }
    }
]
```

3. **Register in available functions**:
```python
available_functions = {
    "calculate": calculate,
    "get_weather": get_weather,
}
```

### Changing the Model

Groq supports various models. To switch:

```python
MODEL = 'llama-3.1-8b-instant'    # Faster, less capable
MODEL = 'llama-3.3-70b-versatile' # Balanced (default)
MODEL = 'mixtral-8x7b-32768'      # Alternative option
```

## 📚 Learning Path

1. **Start Here**: Run `simple_agent.py` to understand basic function calling
2. **Structure**: Run `agent_structure.py` to learn structured responses with Pydantic
3. **Compare**: Understand the differences between simple and structured approaches
4. **Experiment**: Modify functions or add new math/weather operations
5. **WSL Practice**: Try running scripts on WSL for Linux environment experience
6. **Extend**: Add new tools (web search, file operations, database queries)
7. **Advanced**: Implement multi-step reasoning and complex workflows
8. **Production**: Build type-safe agents for real-world applications

## 🛠️ Troubleshooting

### Common Issues

**API Key Error**:
```
Error: GROQ_API_KEY environment variable not set
```
- Solution: Set your API key as shown in Step 4

**Module Not Found**:
```
ModuleNotFoundError: No module named 'groq'
```
- Solution: Ensure virtual environment is activated and groq is installed

**Rate Limiting**:
```
RateLimitError: Rate limit exceeded
```
- Solution: Wait a moment and try again (free tier has limits)

**Invalid Expression**:
```
{"error": "Invalid expression"}
```
- Solution: This is expected behavior for malformed math expressions

**Pydantic Validation Error**:
```
ValidationError: X validation errors for ResponseModel
```
- Solution: The LLM output doesn't match the expected Pydantic model structure

**WSL Path Issues**:
```
FileNotFoundError: [Errno 2] No such file or directory
```
- Solution: Use proper WSL paths (`/mnt/c/...`) or copy files to WSL filesystem

**WSL Environment Variables**:
```
GROQ_API_KEY not found in WSL
```
- Solution: Set environment variables in WSL separately from Windows

### Getting Help

- **Groq Documentation**: [https://console.groq.com/docs](https://console.groq.com/docs)
- **API Reference**: [https://console.groq.com/docs/api-reference](https://console.groq.com/docs/api-reference)
- **Community**: Groq Discord and forums

## 📈 Next Steps

Once you're comfortable with the basic agent:

1. **Explore Multi-Agent Systems**: Build agents that collaborate
2. **Add Memory**: Implement conversation history and context retention
3. **Web Integration**: Connect to APIs and web services
4. **GUI Interface**: Create a chat interface using Streamlit or Gradio
5. **Production Deployment**: Scale your agent for real-world use

## 🤝 Contributing

Feel free to:
- Add new example agents
- Improve existing scripts
- Share your learning modifications
- Report issues or suggest improvements

## 📄 License

This project is for educational purposes. Please respect Groq's terms of service and API usage guidelines.

---

**Happy Agent Building! 🤖**