"""
Step 1 — The ReAct Loop (no API calls)

Goal: understand the exact shape of the message history.
Everything here is hardcoded. Just run it and read the output.
"""

def simulate_react_loop():
    print("=" * 55)
    print("USER: What is 25 * 48, then add 100 to the result?")
    print("=" * 55)

    # This is the list of messages we build up across iterations.
    # In the real agent, we send this whole list to OpenAI every iteration.
    messages = [
        {"role": "system",  "content": "You are a helpful assistant with tools."},
        {"role": "user",    "content": "What is 25 * 48, then add 100?"},
    ]

    # ── ITERATION 1 ──────────────────────────────────────────────────
    print("\n── Iteration 1 ──────────────────────────────────────────")

    # The LLM responds with a thought AND a tool call request
    llm_response_1 = {
        "role": "assistant",
        "content": "I need to multiply 25 * 48 first.",
        "tool_calls": [{
            "id": "call_001",
            "function": {"name": "calculator", "arguments": '{"expression": "25 * 48"}'}
        }]
    }
    messages.append(llm_response_1)

    print(f"🧠 THOUGHT:  {llm_response_1['content']}")
    print(f"🔧 ACTION:   calculator(25 * 48)")

    # Our code runs the calculator and gets a result
    tool_result_1 = {
        "role": "tool",
        "tool_call_id": "call_001",
        "content": "1200"          # this is what our Python calculator returns
    }
    messages.append(tool_result_1)

    print(f"👁  OBSERVE:  {tool_result_1['content']}")
    print(f"   → Does LLM have tool_calls? YES → keep looping")

    # ── ITERATION 2 ──────────────────────────────────────────────────
    print("\n── Iteration 2 ──────────────────────────────────────────")

    llm_response_2 = {
        "role": "assistant",
        "content": "25 * 48 = 1200. Now I add 100.",
        "tool_calls": [{
            "id": "call_002",
            "function": {"name": "calculator", "arguments": '{"expression": "1200 + 100"}'}
        }]
    }
    messages.append(llm_response_2)

    print(f"🧠 THOUGHT:  {llm_response_2['content']}")
    print(f"🔧 ACTION:   calculator(1200 + 100)")

    tool_result_2 = {
        "role": "tool",
        "tool_call_id": "call_002",
        "content": "1300"
    }
    messages.append(tool_result_2)

    print(f"👁  OBSERVE:  {tool_result_2['content']}")
    print(f"   → Does LLM have tool_calls? YES → keep looping")

    # ── ITERATION 3 ──────────────────────────────────────────────────
    print("\n── Iteration 3 ──────────────────────────────────────────")

    llm_response_3 = {
        "role": "assistant",
        "content": "Final answer: 25 * 48 = 1200, and adding 100 gives 1300.",
        # NO tool_calls key — this is the signal the loop ends
    }
    messages.append(llm_response_3)

    print(f"🧠 THOUGHT:  (none)")
    print(f"🔧 ACTION:   (none)")
    print(f"✅ ANSWER:   {llm_response_3['content']}")
    print(f"   → Does LLM have tool_calls? NO → STOP")

    # ── Show the full message history ─────────────────────────────────
    print("\n" + "=" * 55)
    print("FULL MESSAGE HISTORY (sent to OpenAI on every iteration)")
    print("=" * 55)
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id", "")

        print(f"\n[{i}] {role}")
        if content:
            print(f"     content: {content}")
        if tool_calls:
            for tc in tool_calls:
                print(f"     tool_call: {tc['function']['name']}({tc['function']['arguments']})")
        if tool_call_id:
            print(f"     tool_call_id: {tool_call_id}  ← links result back to the request")

    print("\n── The one rule ─────────────────────────────────────────")
    print("while response has tool_calls:")
    print("    run each tool")
    print("    append results to messages")
    print("    call LLM again with full history")
    print()
    print("when response has NO tool_calls → that IS the answer")

simulate_react_loop()