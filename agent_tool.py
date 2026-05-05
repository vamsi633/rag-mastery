"""
Step 2 — Tool Schemas & Execution

Two concepts:
1. Schema = what the LLM sees (JSON description of the tool)
2. Implementation = what Python actually runs

No API calls. We define tools and test them locally.
"""

import json
import math


# ═══════════════════════════════════════════════════════════════════
# PART A: The schema (this is what we send to OpenAI as `tools`)
# ═══════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a math expression. Supports +, -, *, /, "
                "** (power), math.sqrt(), math.pi. "
                "Always use this — never do math in your head."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python math expression, e.g. '25 * 48'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "string_tools",
            "description": (
                "Perform string operations: word_count, reverse, "
                "uppercase, lowercase, char_count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["word_count", "reverse", "uppercase",
                                 "lowercase", "char_count"],
                        "description": "The operation to perform."
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to operate on."
                    }
                },
                "required": ["operation", "text"]
            }
        }
    }
]


# ═══════════════════════════════════════════════════════════════════
# PART B: The implementations (what Python actually runs)
# ═══════════════════════════════════════════════════════════════════

def calculator(expression: str) -> str:
    try:
        allowed = {"math": math, "__builtins__": {}}
        result = eval(expression, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def string_tools(operation: str, text: str) -> str:
    ops = {
        "word_count":  lambda t: f"{len(t.split())} words",
        "reverse":     lambda t: t[::-1],
        "uppercase":   lambda t: t.upper(),
        "lowercase":   lambda t: t.lower(),
        "char_count":  lambda t: f"{len(t)} characters",
    }
    if operation not in ops:
        return f"Unknown operation: {operation}"
    return ops[operation](text)


# ═══════════════════════════════════════════════════════════════════
# PART C: The executor (bridges LLM decisions → Python functions)
# ═══════════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    "calculator": calculator,
    "string_tools": string_tools,
}


def execute_tool(tool_name: str, tool_args_json: str) -> str:
    """
    Given a tool name and JSON string of arguments (both from the LLM),
    find the right function, parse the args, run it, return the result.
    """
    # Parse the JSON arguments
    try:
        args = json.loads(tool_args_json)
    except json.JSONDecodeError:
        return f"Error: couldn't parse arguments: {tool_args_json}"

    # Find the function
    if tool_name not in TOOL_REGISTRY:
        return f"Error: unknown tool '{tool_name}'"

    # Call it
    fn = TOOL_REGISTRY[tool_name]
    try:
        return fn(**args)
    except TypeError as e:
        return f"Error calling {tool_name}: {e}"


# ═══════════════════════════════════════════════════════════════════
# TEST: call each tool manually
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("PART A — Schemas (what the LLM sees)")
    print("=" * 55)
    for tool in TOOLS:
        fn = tool["function"]
        params = list(fn["parameters"]["properties"].keys())
        print(f"\n  Tool: {fn['name']}")
        print(f"  Params: {params}")
        print(f"  Description: {fn['description'][:60]}...")

    print("\n" + "=" * 55)
    print("PART B — Execute tools manually")
    print("=" * 55)

    # Simulating what happens when the LLM says "call calculator"
    tests = [
        ("calculator",    '{"expression": "25 * 48"}'),
        ("calculator",    '{"expression": "math.sqrt(144)"}'),
        ("calculator",    '{"expression": "2 ** 10"}'),
        ("string_tools",  '{"operation": "word_count", "text": "the quick brown fox"}'),
        ("string_tools",  '{"operation": "reverse", "text": "hello"}'),
        ("string_tools",  '{"operation": "uppercase", "text": "agents are cool"}'),
        ("unknown_tool",  '{}'),
    ]

    for tool_name, args_json in tests:
        result = execute_tool(tool_name, args_json)
        print(f"\n  {tool_name}({args_json})")
        print(f"  → {result}")

    print("\n" + "=" * 55)
    print("KEY INSIGHT")
    print("=" * 55)
    print("The LLM produces TWO things: a tool NAME and ARGUMENTS (as JSON).")
    print("Your execute_tool() reads those, finds the Python function, runs it.")
    print("The LLM never touches Python. It just generates structured text.")