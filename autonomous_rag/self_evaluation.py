from dotenv import load_dotenv
from openai import OpenAI
import json
import os

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def self_evaluate(goal:str,completed_work:dict)->dict:
     """
    The agent evaluates its own output.

    Inputs:
      - goal: what was requested
      - completed_work: dict of task results

    Returns:
      - quality_score: 1-10
      - is_complete: bool
      - missing: list of gaps
      - additional_tasks: list of new tasks to fill gaps
    """
     work_summary="\n".join(
          f"Task: {task} \nResult: {result[:300]}" for task,result in completed_work.items()
     )
     content="""You are a quality reviewer. Evaluate whether the completed work fully achieves the original goal
     
     Be critical.Check for:
     1.Completeness - Does it cover ALL aspects of the goal?
     2.Accuracy - are the facts and numbers conistsent?
     3.Actionability - are there concrete recommendations, not just data?
     4.Gaps - what import information is missing?
     
     Respond with JSON:
     {
     "quality_score":1-10,
     "is_complete":true/false,
     "strengths":["What was done well"],
     "gaps":["what is missing or weak"],
     "additional_tasks":[{"description":"task to fill the gap","tool":"tool_name"}]
     }
     """


     response=client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
               {"role":"system","content":content},
               {"role":"user","content":f"Goal: {goal} \n\nCOMPLETED WORK:\n{work_summary}"},
          ],
          temperature=0,
          response_format={"type":"json_object"},
     )

     return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    goal = "Prepare a quarterly business review for the leadership team"

    # Simulate INCOMPLETE work — missing some key data
    incomplete_work = {
        "Get financial data": "Q3 revenue was $38.2M, up 14.7% from Q3 2023. Net profit margin improved to 31.2%.",
        "Get sales by region": "North America: $14.2M, Europe: $8.8M, Asia Pacific: $11.3M, Latin America: $3.9M",
        "Write summary": "NovaTech had a strong Q3 with $38.2M in revenue. All regions grew.",
    }

    print(f"🎯 GOAL: {goal}")
    print(f"\n📋 COMPLETED WORK (intentionally incomplete):")
    for task, result in incomplete_work.items():
        print(f"  • {task}: {result[:80]}...")

    print(f"\n{'='*60}")
    print("🔍 SELF-EVALUATION:")
    print(f"{'='*60}")

    evaluation = self_evaluate(goal, incomplete_work)

    print(f"\n  Quality Score: {evaluation['quality_score']}/10")
    print(f"  Complete: {evaluation['is_complete']}")

    print(f"\n  Strengths:")
    for s in evaluation.get("strengths", []):
        print(f"    ✅ {s}")

    print(f"\n  Gaps:")
    for g in evaluation.get("gaps", []):
        print(f"    ❌ {g}")

    if evaluation.get("additional_tasks"):
        print(f"\n  Additional tasks needed:")
        for task in evaluation["additional_tasks"]:
            desc = task.get("description", "")
            tool = task.get("tool", "")
            print(f"    → {desc} (tool: {tool})")

    # Now simulate COMPLETE work
    print(f"\n\n{'='*60}")
    print("NOW WITH MORE COMPLETE WORK:")
    print(f"{'='*60}")

    complete_work = {
        "Get financial data": "Q3 revenue was $38.2M, up 14.7% from Q3 2023. Net profit margin 31.2%, up from 28.4% in Q2. Cash reserves: $52.8M.",
        "Get sales by region": "North America: $14.2M (+14.5%), Europe: $8.8M (+8.6%), Asia Pacific: $11.3M (+16.5%), Latin America: $3.9M (+21.9%)",
        "Get employee data": "847 employees across 6 offices. Engineering: 358, Sales: 127 new deals closed. Employee satisfaction: 4.3/5. Turnover: 8.2% (below industry 13.5%).",
        "Get support metrics": "NPS: 72 (up from 68). Churn: 2.1% (down from 2.8%). Avg ticket resolution: 4.2 hours.",
        "Get risk factors": "Competition from major cloud providers, currency exposure (EUR/USD, JPY/USD), AWS dependency (73%), tight engineering talent market.",
        "Get strategic priorities": "CloudSync Enterprise expansion to 500 customers, Singapore data center phase 2, AI integration beta in December, 99.99% uptime SLA target.",
        "Write report": "Comprehensive Q3 review covering financials, team growth, customer health, risks, and Q4 priorities. Recommends increased Asia Pacific investment and accelerated AI roadmap.",
    }

    evaluation2 = self_evaluate(goal, complete_work)

    print(f"\n  Quality Score: {evaluation2['quality_score']}/10")
    print(f"  Complete: {evaluation2['is_complete']}")

    print(f"\n  Strengths:")
    for s in evaluation2.get("strengths", []):
        print(f"    ✅ {s}")

    print(f"\n  Gaps:")
    for g in evaluation2.get("gaps", []):
        print(f"    ❌ {g}")

    if evaluation2.get("additional_tasks"):
        print(f"\n  Additional tasks:")
        for task in evaluation2["additional_tasks"]:
            print(f"    → {task.get('description', '')} (tool: {task.get('tool', '')})")
    else:
        print(f"\n  No additional tasks needed! ✅")

    print(f"""
{'='*60}
WHAT THIS SHOWS:
{'='*60}

Incomplete work → Low score, identified gaps, suggested new tasks
Complete work   → High score, approved, no additional tasks

The agent can JUDGE ITS OWN QUALITY and decide:
  Score < 7? → Add more tasks and keep going
  Score >= 7? → Good enough, deliver the report

This self-critique loop is what separates autonomous from agentic.
""")