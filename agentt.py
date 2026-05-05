"""
Step 3 — The Real ReAct Agent (first live API call)

This wires together:
  - The loop from step 1
  - The tools from step 2
  - Real OpenAI API calls

~40 lines of actual logic. That's all an agent is.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Import tools from step 2
from agent_tool import TOOLS, TOOL_REGISTRY, execute_tool

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
- Always use the calculator for math — never compute in your head.
- Think step by step before calling a tool.
- When you have the final answer, just say it clearly."""


def run_agent(question: str, max_iterations: int = 10):
    """The complete ReAct loop. That's it. This is the whole agent."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    print(f"\n{'='*55}")
    print(f"🤔 QUESTION: {question}")
    print(f"{'='*55}")

    for i in range(max_iterations):
        print(f"\n── Iteration {i+1} ─────────────────────────────────")

        # ① Call the LLM with messages + tools
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",    # LLM decides: use tool or answer
        )

        msg = response.choices[0].message

        # Always append the assistant's response to history
        messages.append(msg)

        # Print any reasoning the LLM shared
        if msg.content:
            print(f"🧠 THOUGHT: {msg.content}")

        # ② Check: did the LLM request tool calls?
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"🔧 ACTION:  {tc.function.name}({tc.function.arguments})")

                # ③ Execute the tool
                result = execute_tool(tc.function.name, tc.function.arguments)
                print(f"👁  OBSERVE: {result}")

                # ④ Feed the result back into the message history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # Loop continues — LLM gets to see the result next iteration

        else:
            # No tool calls → LLM is done
            print(f"\n✅ FINAL ANSWER: {msg.content}")
            print(f"   Iterations: {i+1}")
            print(f"   Messages in history: {len(messages)}")
            print(f"   Tokens used: {response.usage.total_tokens}")
            return msg.content

    return "Hit max iterations without answering."


if __name__ == "__main__":
    # TEST 1: single tool call
    run_agent("What is 847 * 23?")

    # TEST 2: multi-step reasoning (needs 2+ tool calls)
    run_agent(
        "I have 15 boxes with 48 items each. "
        "I give away 127 items. How many are left?"
    )

    # TEST 3: uses a different tool
    run_agent("How many words are in: 'to be or not to be that is the question'?")

    # TEST 4: no tools needed at all
    run_agent("What is the capital of Japan?")