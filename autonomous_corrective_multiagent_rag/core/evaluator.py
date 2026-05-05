import json
from config.settings import client,LLM_MODEL


system_prompt="""
Evaluate if the work achieves the goal. Be critical but realistic.

Only suggest additional tasks using these tools:
-search_documents:Search McKinsey report
-query_database:SQL on PostgreSQL (e-commerce tables)
-calculate:Math

Do NOT suggest task requiring data that doesn't exist.
Score 7+ means good enough to deliver.

Respond in JSON:
{
"quality_score":1-10,
"is_completed":true/false,
"gaps":["what is missing"],
"additional_tasks":[{"description":"...","tool":"search_documents/query_database/calculate"}]
}
"""
def evaluate(goal,completed_results):
    """Score the completed work against the original goal.
    Returns: {"quality_score":1-10,"is_completed":bool,"gaps":[...],"additional_tasks":[...]}"""

    work_summary="\n".join(
        f"{k}:{v[:300]}..." for k,v in completed_results.items()
    )

    response=client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":'user',"content":f"GOAL:{goal}\n\nWORK:\n{work_summary}"},
        ]
    )