"""
ARCHITECTURE DIAGRAM: AI Agent Framework

Visual representation of component relationships and data flow.
"""

# ============================================================================
# SYSTEM ARCHITECTURE
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER APPLICATION                              │
│                                                                         │
│  agent = Agent(llm_provider, tool_registry, agent_config)             │
│  state = agent.run("User query")                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        AGENT LOOP (core/agent.py)                      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │ 1. Initialize: Add system prompt + user query to memory     │     │
│  │ 2. Loop (max_iterations):                                   │     │
│  │    a. Call LLM with conversation history                    │     │
│  │    b. Parse reasoning JSON                                  │     │
│  │    c. Decide: Tool? Respond? Error?                         │     │
│  │       ├─ Tool: Execute → Add observation                    │     │
│  │       └─ Respond: Return final answer                       │     │
│  │ 3. Return AgentState with trace + final response            │     │
│  └──────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
         ▲                      ▲                      ▲
         │                      │                      │
         │                      │                      │
    ┌────┴──────┐          ┌────┴──────┐       ┌──────┴────┐
    │  (Calls)   │          │ (Calls)   │       │ (Updates) │
    ▼            ▼          ▼           ▼       ▼            ▼

┌──────────────────┐  ┌───────────────────┐  ┌──────────────────────┐
│  LLM PROVIDER    │  │  TOOL REGISTRY    │  │   MEMORY SYSTEM      │
│  (core/llm...)   │  │  (core/tool.py)   │  │  (core/memory.py)    │
│                  │  │                   │  │                      │
│ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌────────────────┐   │
│ │GroqProvider │ │  │ │ToolRegistry  │ │  │ │Short-term:    │   │
│ │ - generate()│◄┼──┼─┤ - register() │ │  │ │ Conv. History │   │
│ └──────────────┘ │  │ ├─────────────┤ │  │ └────────────────┘   │
│ ┌──────────────┐ │  │ │ Tools:      │ │  │ ┌────────────────┐   │
│ │OllamaProvider│ │  │ │ - calculate │ │  │ │Long-term:     │   │
│ │              │ │  │ │ - search    │ │  │ │ Observations  │   │
│ └──────────────┘ │  │ │ - get_fact  │ │  │ └────────────────┘   │
│ ┌──────────────┐ │  │ └─────────────┘ │  │ ┌────────────────┐   │
│ │CustomProvider│ │  │ execute(tool)→ │  │ │Methods:        │   │
│ │(user-defined)│ │  │ {"result": ...} │  │ │- add_message() │   │
│ └──────────────┘ │  │                 │  │ │- add_obs()     │   │
│                  │  │ to_openai_fmt()│  │ │- get_hist()    │   │
│ Returns:         │  │ ─────────────► │  │ │- get_obs()     │   │
│ LLMResponse      │  │ [{func schema}] │  │ └────────────────┘   │
└──────────────────┘  └───────────────────┘  └──────────────────────┘
       ▲                       ▲                       ▲
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               │
                        (Data exchange)
"""

# ============================================================================
# DATA FLOW: SINGLE AGENT ITERATION
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────┐
│                     ITERATION START                                 │
│                 User Query: "What is 25*4?"                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: PREPARE LLM INPUT                                         │
│  - Get conversation history from Memory                            │
│  - Convert to LLMMessage format                                    │
│  - Get tools in OpenAI format                                      │
│  - Build system prompt with tool descriptions                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: CALL LLM PROVIDER                                         │
│  llm.generate(messages, tools, temperature, max_tokens)           │
│                                                                    │
│  Groq API Request:                                                │
│  POST https://api.groq.com/openai/v1/chat/completions            │
│  {                                                                │
│    "model": "llama-3.3-70b-versatile",                            │
│    "messages": [...],                                             │
│    "tools": [...],                                                │
│    "temperature": 0.7                                             │
│  }                                                                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: RECEIVE LLM RESPONSE                                      │
│  {                                                                 │
│    "content": "{\"thought\": \"Need to calculate\",               │
│                \"action\": \"tool\",                              │
│                \"tool_name\": \"calculate\",                      │
│                \"tool_input\": {\"expression\": \"25*4\"}}",      │
│    "tool_calls": [...]                                            │
│  }                                                                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: PARSE REASONING                                           │
│  Extract JSON from response:                                       │
│  {                                                                 │
│    "thought": "Need to calculate 25*4",                           │
│    "action": "tool",                                              │
│    "tool_name": "calculate",                                      │
│    "tool_input": {"expression": "25*4"}                           │
│  }                                                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: DECIDE AND ACT                                            │
│                                                                    │
│  Action = "tool" ?                                                │
│     ├─ YES ─────────────────────┐                                 │
│     │                            │                                 │
│     └──────► EXECUTE TOOL        │                                 │
│              - Tool: calculate   │                                 │
│              - Input: expression │                                 │
│              - Output: {"result":│ "100"}                          │
│                                  │                                 │
│                          Add observations to memory                │
│                          Add assistant message to memory           │
│                          Add tool result to memory                 │
│                                  │                                 │
│                          Continue to next iteration                │
│                                                                    │
│     ├─ NO ───────────────────────────────────────────┐            │
│     │                                                │            │
│     └──────► RETURN FINAL RESPONSE                   │            │
│              - Extract from reasoning                │            │
│              - Set state to "completed"              │            │
│              - Break from loop                       │            │
│                                                      │            │
└──────────────────────────────────────────────────────┼──────────────┘
                                                       │
                              ┌────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ITERATION END / RESULT                           │
│                                                                    │
│  AgentState:                                                       │
│  {                                                                 │
│    "iteration": 2,                                                │
│    "status": "completed",                                         │
│    "final_response": "25 * 4 equals 100",                        │
│    "tool_calls": 1,                                               │
│    "reasoning_trace": [{...}, {...}],                             │
│    "errors": []                                                   │
│  }                                                                 │
└─────────────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# TOOL EXECUTION FLOW
# ============================================================================

"""
┌──────────────────────────┐
│  Agent decides to use    │
│  tool: "calculate"       │
└──────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────┐
│  Tool Registry: tools.execute("calculate", ...)      │
└──────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────┐
│  1. Get tool: tool = registry.get("calculate")      │
│  2. Call tool.execute(**kwargs)                     │
│  3. Return: {"result": ..., "error": None}          │
└──────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────┐
│  Tool Execution:                                    │
│                                                     │
│  FunctionTool(func=calculate)                      │
│    │                                                │
│    └─ execute(expression="25*4")                   │
│       │                                             │
│       └─ result = calculate("25*4")                │
│          (calls eval with safe dict)               │
│       │                                             │
│       └─ returns {"result": "100", "error": None}  │
└──────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────┐
│  Agent:                                             │
│  1. Receive result                                  │
│  2. Add to memory:                                  │
│     - Message: "assistant" (LLM thought)          │
│     - Message: "tool" (tool result)                │
│     - Observation: {"type": "tool", ...}           │
│  3. Continue to next iteration                     │
└──────────────────────────────────────────────────────┘
"""

# ============================================================================
# MEMORY STRUCTURE
# ============================================================================

"""
Memory Class:
┌──────────────────────────────────────────────────────────┐
│  Short-term: list[dict]                                 │
│  ┌────────────────────────────────────────────────────┐ │
│  │ [                                                  │ │
│  │   {"role": "system", "content": "You are..."},    │ │
│  │   {"role": "user", "content": "What is 25*4?"},   │ │
│  │   {"role": "assistant", "content": "{json}"},     │ │
│  │   {"role": "tool", "content": "{\"result\": ...}"},│ │
│  │   ...                                              │ │
│  │ ]                                                  │ │
│  │                                                    │ │
│  │ Max size: max_history (default 100)               │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  Long-term: list[Observation]                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ [                                                  │ │
│  │   Observation(                                     │ │
│  │     type="tool",                                  │ │
│  │     content="{\"result\": \"100\"}",             │ │
│  │     tool_name="calculate",                       │ │
│  │     timestamp=2025-01-03T10:30:45,               │ │
│  │     metadata={"input": {"expression": "25*4"}}   │ │
│  │   ),                                              │ │
│  │   ...                                              │ │
│  │ ]                                                  │ │
│  │                                                    │ │
│  │ Unlimited (for full history)                      │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
"""

# ============================================================================
# CLASS HIERARCHY
# ============================================================================

"""
Tool (ABC) ──┬─ FunctionTool (wraps functions)
             └─ CustomTool (user-defined classes)

LLMProvider (ABC) ──┬─ GroqProvider
                    ├─ OllamaProvider
                    └─ CustomProvider (user-defined)

Agent ──┬─ uses: LLMProvider
        ├─ uses: ToolRegistry
        └─ uses: Memory

AgentConfig ─ configures: Agent
AgentState ─ output from: Agent.run()
"""

# ============================================================================
# CONFIGURATION HIERARCHY
# ============================================================================

"""
Agent Configuration Chain:

User Code
  │
  ├─ Provides: LLMProvider
  │            ├─ model name
  │            ├─ api_key
  │            └─ temperature
  │
  ├─ Provides: ToolRegistry
  │            ├─ Tool 1
  │            ├─ Tool 2
  │            └─ Tool N
  │
  └─ Provides: AgentConfig
               ├─ max_iterations
               ├─ verbose
               ├─ temperature
               └─ system_prompt

                  │
                  ▼
            Agent Instance
            (combines all)
                  │
                  ├─ memory: Memory
                  ├─ state: AgentState
                  └─ methods:
                     ├─ run(query)
                     ├─ reset()
                     └─ get_memory_state()
"""

# ============================================================================
# MESSAGE FORMAT FLOW
# ============================================================================

"""
User Input: "What is 25 * 4?"
                  │
                  ▼
Memory Format (short-term):
  {"role": "user", "content": "What is 25 * 4?"}
                  │
                  ▼
LLMMessage Format (for LLM):
  LLMMessage(role="user", content="What is 25 * 4?")
                  │
                  ▼
LLM Provider Input:
  - List of LLMMessage
  - List of tools (OpenAI format)
  - Temperature, max_tokens, etc.
                  │
                  ▼
LLM Provider Output (LLMResponse):
  LLMResponse(
    content="{\"thought\": ..., \"action\": ...}",
    tool_calls=[{"name": "calculate", "arguments": "..."}]
  )
                  │
                  ▼
Agent Parsing:
  - Extract JSON from content
  - Parse tool_calls
  - Decide action
                  │
                  ▼
Memory Update:
  short_term.append({"role": "assistant", "content": ...})
  long_term.append(Observation(...))
"""

# ============================================================================
# ERROR HANDLING FLOW
# ============================================================================

"""
During Agent.run():

Try:
  1. Get LLM response
  2. Parse reasoning
  3. Execute tool (if applicable)

Except:
  ├─ LLM error
  │  └─ state.status = "error"
  │  └─ state.errors.append(error_msg)
  │  └─ break loop
  │
  ├─ Parse error
  │  └─ fallback: treat response as "respond"
  │  └─ continue or finish
  │
  └─ Tool error
     └─ captured in tool result
     └─ added to conversation
     └─ continue loop
"""

# ============================================================================
# VERBOSE OUTPUT EXAMPLE
# ============================================================================

"""
============================================================
Agent Starting: What is 25 * 4?
============================================================

Iteration 1/10
Thought: I need to calculate 25 * 4
Action: tool
Tool: calculate
Input: {
  "expression": "25 * 4"
}
Result: {'result': '100', 'error': None}

Iteration 2/10
Thought: I have the result, now I can provide the answer
Action: respond
Response: The result of 25 * 4 is 100.

============================================================
Final Status: completed
Total Iterations: 2
Tool Calls: 1
Final Response: The result of 25 * 4 is 100.
============================================================
"""
