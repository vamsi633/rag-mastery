"""
core/planner.py — Decompose goals into ordered sub-tasks.
"""

import json
from config.settings import client, LLM_MODEL
from tools.sql_tool import get_schema


def plan(goal):
    """Break a goal into concrete, ordered sub-tasks."""
    schema = get_schema()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"""Break the goal into 4-7 sub-tasks.

Available tools:
- search_documents: Search McKinsey Technology Trends Outlook 2024 report
- query_database: SQL on PostgreSQL
- calculate: Math
- synthesize: Write the final report (depends on all data tasks)

DATABASE SCHEMA:
{schema[:3000]}

RULES:
- Use ONLY columns from the schema above
- Use PostgreSQL syntax
- Final task must be synthesize, depending on all others
- Each task should be specific enough for a single tool call

Respond with JSON:
{{"tasks": [{{"id": 1, "description": "...", "tool": "...", "depends_on": []}}]}}"""
            },
            {"role": "user", "content": f"Goal: {goal}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content).get("tasks", [])