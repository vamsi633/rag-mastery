"""
agents/base_agent.py — Generic agent runner.
Any specialist agent uses this same loop.
"""

import json
from config.settings import client, LLM_MODEL, MAX_AGENT_STEPS


def run(system_prompt, tools, tool_map, question, agent_name="agent", max_steps=MAX_AGENT_STEPS):
    """
    Generic agent loop. Works for ANY specialist.
    
    Args:
        system_prompt: The specialist's focused prompt
        tools: Tool definitions (OpenAI format)
        tool_map: {tool_name: callable}
        question: What to answer
        agent_name: For logging
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    tools_used = []

    print(f"\n  🤖 [{agent_name.upper()}] {question[:70]}...")

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
            temperature=0,
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                tools_used.append(func_name)

                print(f"     Step {step+1}: 🔧 {func_name}")
                for k, v in args.items():
                    print(f"              {k}: {str(v)[:80]}")

                func = tool_map.get(func_name)
                if func:
                    try:
                        result = func(**args)
                    except TypeError:
                        result = func()
                else:
                    result = f"Unknown tool: {func_name}"

                preview = result[:120].replace("\n", " ")
                print(f"              → {preview}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
        else:
            print(f"     ✅ Done ({step+1} steps, tools: {tools_used})")
            return message.content

    return "Agent reached max steps."