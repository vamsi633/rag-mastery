"""
main.py — Entry point for the unified system.
Handles both questions (route → agents) and goals (plan → execute → evaluate).
"""

import sys
import json
from config.settings import index, DATABASE_URL
from core.router import route
from core.grader import grade_and_filter
from core.synthesizer import synthesize
from core.planner import plan
from core.evaluator import evaluate
from agents.registry import run_agent, AGENTS
from tools.search_tool import search_raw
from tools.memory_tool import save as save_memory, recall as recall_memory
import psycopg2


def handle_question(question):
    """Route a question to specialist agent(s), grade docs, synthesize."""
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}")

    # Route
    routing = route(question)
    agents = routing.get("agents", ["research"])
    sub_questions = routing.get("sub_questions", {})
    print(f"\n  📡 ROUTER: {agents} — {routing.get('reasoning', '')}")

    # Run each specialist
    agent_results = {}
    for agent_name in agents:
        if agent_name not in AGENTS:
            print(f"  ⚠️  Unknown agent: {agent_name}, skipping")
            continue
        sub_q = sub_questions.get(agent_name, question)
        result = run_agent(agent_name, sub_q)
        agent_results[agent_name] = result

    # Synthesize if multiple agents
    if len(agent_results) == 1:
        answer = list(agent_results.values())[0]
    else:
        print(f"\n  📝 SYNTHESIZER: Combining {len(agent_results)} agent results...")
        answer = synthesize(question, agent_results)

    # Save to memory
    save_memory(question, answer, ",".join(agents), json.dumps(list(agent_results.keys())))

    return answer


def handle_goal(goal):
    """Plan → execute → evaluate → refine for complex goals."""
    from config.settings import MAX_AUTONOMOUS_ITERATIONS

    print(f"\n{'='*60}")
    print(f"🎯 GOAL: {goal}")
    print(f"{'='*60}")

    # Plan
    print(f"\n📋 PLANNING...")
    tasks = plan(goal)
    for t in tasks:
        deps = t.get("depends_on", [])
        deps_str = f" (after {deps})" if deps else ""
        print(f"  Task {t['id']}: [{t.get('tool', 'synthesize')}] {t['description'][:70]}{deps_str}")

    completed = {}
    final_report = ""

    for iteration in range(MAX_AUTONOMOUS_ITERATIONS):
        print(f"\n{'─'*40} Iteration {iteration+1} {'─'*40}")

        # Execute tasks
        progress = True
        while progress:
            progress = False
            for task in tasks:
                key = f"task_{task['id']}"
                if key in completed:
                    continue
                deps = task.get("depends_on", [])
                if not all(f"task_{d}" in completed for d in deps):
                    continue

                progress = True
                tool = task.get("tool", "synthesize")
                desc = task["description"]

                print(f"\n  ⚡ Task {task['id']}: {desc[:60]}...")

                if tool == "synthesize":
                    from config.settings import client, LLM_MODEL
                    all_data = "\n\n".join(f"--- {k} ---\n{v}" for k, v in completed.items())
                    resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": "Write a clear, actionable report with specific numbers and recommendations."},
                            {"role": "user", "content": f"Task: {desc}\n\nData:\n{all_data}"},
                        ],
                        temperature=0,
                    )
                    result = resp.choices[0].message.content
                    final_report = result
                elif tool == "search_documents":
                    # Use grading for document search
                    chunks = search_raw(desc)
                    action, filtered, all_graded = grade_and_filter(desc, chunks)
                    print(f"     CRAG: {action} ({len(filtered)}/{len(chunks)} chunks kept)")
                    if filtered:
                        result = "\n".join(c["text"] for c in filtered)
                    else:
                        result = "No relevant documents found."
                elif tool == "query_database":
                    result = run_agent("sales", desc)
                elif tool == "calculate":
                    result = run_agent("sales", desc)
                else:
                    result = run_agent("research", desc)

                completed[key] = result
                preview = result[:120].replace("\n", " ")
                print(f"     → {preview}...")

        # Evaluate
        print(f"\n  🔍 EVALUATING...")
        evaluation = evaluate(goal, completed)
        score = evaluation.get("quality_score", 0)
        gaps = evaluation.get("gaps", [])
        new_tasks = evaluation.get("additional_tasks", [])

        print(f"     Score: {score}/10")
        for g in gaps:
            print(f"     Gap: {g}")

        if score >= 7:
            print(f"\n  ✅ Quality met! Delivering.")
            break

        if not new_tasks:
            print(f"\n  ⚠️  No new tasks suggested. Delivering best effort.")
            break

        # Add refinement tasks
        max_id = max(t["id"] for t in tasks)
        for i, nt in enumerate(new_tasks):
            new_id = max_id + i + 1
            nt["id"] = new_id
            nt["depends_on"] = nt.get("depends_on", [])
            tasks.append(nt)
            print(f"     New Task {new_id}: [{nt.get('tool', 'search_documents')}] {nt['description'][:60]}")

        synth_id = max_id + len(new_tasks) + 1
        tasks.append({
            "id": synth_id,
            "description": "Re-synthesize into improved report",
            "tool": "synthesize",
            "depends_on": [t["id"] for t in tasks if t.get("tool") != "synthesize"],
        })

    save_memory(goal, final_report[:2000], "autonomous", "planner+agents+evaluator")
    return final_report


def main():
    # Verify connections
    stats = index.describe_index_stats()
    print(f"Pinecone: {stats.total_vector_count} vectors")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM olist_orders_dataset")
        print(f"PostgreSQL: {cursor.fetchone()[0]:,} orders")
        conn.close()
    except Exception as e:
        print(f"PostgreSQL error: {e}")
        return

    print(f"\n🤖 E-Commerce Intelligence Assistant")
    print("   Ask a QUESTION or give a GOAL")
    print("   Start with 'goal:' for autonomous mode")
    print("   Type 'memory' to see past conversations")
    print("   Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "memory":
            print(recall_memory())
            continue

        if user_input.lower().startswith("goal:"):
            goal = user_input[5:].strip()
            report = handle_goal(goal)
            print(f"\n{'='*60}")
            print("📄 FINAL REPORT")
            print(f"{'='*60}")
            print(report)
        else:
            answer = handle_question(user_input)
            print(f"\n💬 {answer}")

        print()


if __name__ == "__main__":
    main()