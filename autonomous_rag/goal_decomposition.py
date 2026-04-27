from dotenv import load_dotenv
from openai import OpenAI
import json
import os

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def decompose_goal(goal:str)->list[dict]:
    """Take a high-level goal -> return a list of concrete tasks.
    
    Each task has:
        -description:what to do
        -tool:which tool to use
        -depends_on:which task must complete first

    The LLM is acting as a project manager here
    it doesn't execute anything - it just plans.
    """

    content="""You are a task planner for a business analyst AI.
    Available tools:
    -search_documents:search company reports and handbooks
    -query_database:Run SQL on employee/sales/ticket data
    -calculate:Do math
    -synthesize:Combine findings into a report (no tool needed, you write it)

    Break the goal into 4-7 concrete sub-tasks. Each task should be specific enough that a single tool call can accomplish it.

    Respond with JSON:{"tasks":[{"id":1,"description":"...","tool":"...","depends_on":[]}]}
    depends_on lists task ids that must complete before this task can start.
    the final synthesis task should depend on all data-gathering tasks."""
    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":content},
            {"role":"user","content":f"Goal: {goal}"},
        ],
        temperature=0,
        response_format={"type":"json_object"},
    )

    result=json.loads(response.choices[0].message.content)
    return result.get("tasks",[])


if __name__ == "__main__":
    goals = [
        "Prepare a quarterly business review for the leadership team",
        "Analyze whether we should expand our Asia Pacific sales team",
        "Create a report on employee retention risks and recommendations",
    ]

    for goal in goals:
        print(f"\n{'='*60}")
        print(f"🎯 GOAL: {goal}")
        print(f"{'='*60}")

        tasks = decompose_goal(goal)

        for task in tasks:
            task_id = task.get("id", "?")
            desc = task.get("description", "")
            tool = task.get("tool", "unknown")
            deps = task.get("depends_on", [])

            deps_str = f" ← waits for task {deps}" if deps else " ← can start immediately"
            print(f"\n  Task {task_id}: {desc}")
            print(f"    Tool: {tool}{deps_str}")