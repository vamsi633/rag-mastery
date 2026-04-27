from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
import chromadb
import json
import os
from pinecone import Pinecone


load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_URL=os.getenv("DATABASE_URL")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))



def search_documents(query:str)->str:
    query_emb=client.embeddings.create(model="text-embedding-3-small",input=query).data[0].embedding
    results=index.query(vector=query_emb,top_k=4,include_metadata=True)
    output=""
    for i,match in enumerate(results["matches"]):
        text=match["metadata"]["text"]
        source=match["metadata"].get("source","unknown")
        page=match["metadata"].get("page","?")
        score=round(match["score"],3)
        output+=f"[{source} p{page}, score:{score}] {text}\n\n"
    return output if output else "No results found"


def query_database(sql: str) -> str:
    """Execute read-only SQL on PostgreSQL."""
    sql_upper = sql.upper().strip()
    blocked = ["DROP", "DELETE", "UPDATE", "ALTER", "INSERT", "TRUNCATE"]
    for kw in blocked:
        if kw in sql_upper:
            return f"ERROR: {kw} not allowed."
    if not sql_upper.startswith("SELECT"):
        return "ERROR: Only SELECT allowed."

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = '5000'")
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "No results."
        result = f"Columns: {columns}\n"
        for row in rows[:15]:
            result += str(dict(zip(columns, row))) + "\n"
        return result
    except Exception as e:
        return f"SQL Error: {e}"

def calculate(expression: str) -> str:
    """Safe math calculator."""
    try:
        allowed = set("0123456789.+-*/() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        return str(round(eval(expression, {"__builtins__": {}}, {}), 2))
    except Exception as e:
        return f"Error: {e}"


TOOL_MAP = {
    "search_documents": search_documents,
    "query_database": query_database,
    "calculate": calculate,
}

def get_schema_info() -> str:
    """Get actual table structure so the LLM writes correct SQL."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name IN ('employees', 'sales', 'tickets')
            ORDER BY table_name, ordinal_position
        """)
        rows = cursor.fetchall()

        schema = {}
        for table, col, dtype in rows:
            if table not in schema:
                schema[table] = []
            schema[table].append(f"{col} ({dtype})")

        output = ""
        for table, cols in schema.items():
            output += f"TABLE {table}: {', '.join(cols)}\n"

            cursor.execute(f"SELECT * FROM {table} LIMIT 2")
            col_names = [d[0] for d in cursor.description]
            for row in cursor.fetchall():
                output += f"  Sample: {dict(zip(col_names, row))}\n"

            # Show distinct values for text columns
            for col, dtype in [(c.split(" (")[0], c.split("(")[1].rstrip(")")) for c in cols]:
                if dtype == "text":
                    cursor.execute(f'SELECT DISTINCT "{col}" FROM {table} LIMIT 10')
                    vals = [r[0] for r in cursor.fetchall() if r[0]]
                    if vals and len(vals) <= 10:
                        output += f"  Distinct {col}: {vals}\n"
            output += "\n"

        conn.close()
        return output
    except Exception as e:
        return f"Schema error: {e}"


# Cache it so we don't query schema on every task
_schema_cache = None

def get_cached_schema():
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = get_schema_info()
    return _schema_cache




def plan(goal: str) -> list[dict]:
    schema = get_cached_schema()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""You are a task planner. Break the goal into 4-7 sub-tasks.

Available tools:
- search_documents: Search company reports and handbooks
- query_database: SQL on PostgreSQL
- calculate: Math operations (use ONLY numbers and operators like +, -, *, /)
- synthesize: Combine findings into a final report

ACTUAL DATABASE SCHEMA (use ONLY these columns):
{schema}

IMPORTANT:
- Only query columns that ACTUALLY EXIST in the schema above
- Use PostgreSQL syntax (CURRENT_DATE - INTERVAL '6 months', not DATE_SUB)
- Use double quotes around column names in SQL
- The final task must be synthesize, depending on all other tasks

Respond with JSON:
{{"tasks": [{{"id": 1, "description": "...", "tool": "...", "depends_on": []}}]}}"""
            },
            {"role": "user", "content": f"Goal: {goal}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content).get("tasks", [])


def execute_task(task: dict, completed_results: dict) -> str:
    tool = task.get("tool", "synthesize")
    description = task.get("description", "")

    # Build context from dependencies
    context = ""
    for dep_id in task.get("depends_on", []):
        dep_key = f"task_{dep_id}"
        if dep_key in completed_results:
            context += f"\n{completed_results[dep_key]}\n"

    # Synthesis — combine all results
    if tool == "synthesize":
        all_results = "\n\n".join(
            f"--- {k} ---\n{v}" for k, v in completed_results.items()
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Write a clear, actionable business report with specific numbers, trends, and concrete recommendations. If some data was unavailable, note it and work with what you have. Organize with clear sections."
                },
                {
                    "role": "user",
                    "content": f"Task: {description}\n\nGathered data:\n{all_results}"
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    # Tool-based tasks — LLM generates the specific query
    schema = get_cached_schema()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Generate a {tool} call to accomplish this task.

ACTUAL DATABASE SCHEMA (use ONLY these columns):
{schema}

If search_documents: respond with {{"query": "focused search query"}}
If query_database: respond with {{"sql": "SELECT ..."}}
  - Use ONLY columns from the schema above
  - Use PostgreSQL syntax
  - Use double quotes around column names
  - Use single quotes for string values
If calculate: respond with {{"expression": "numbers and operators ONLY like 38.2 / 33.3 * 100"}}
  - ONLY numbers and +, -, *, / operators. NO text, NO variable names.

Respond ONLY with JSON."""
            },
            {
                "role": "user",
                "content": f"Task: {description}\n\nContext from prior tasks:\n{context}"
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    args = json.loads(response.choices[0].message.content)

    func = TOOL_MAP.get(tool)
    if not func:
        return f"Unknown tool: {tool}"

    try:
        if "query" in args:
            return func(args["query"])
        elif "sql" in args:
            return func(args["sql"])
        elif "expression" in args:
            return func(args["expression"])
        return "No valid arguments generated."
    except Exception as e:
        return f"Tool execution error: {e}"

def evaluate(goal: str, completed_results: dict) -> dict:
    schema = get_cached_schema()

    work_summary = "\n".join(
        f"• {k}: {v[:300]}..." for k, v in completed_results.items()
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Evaluate if the work achieves the goal. Be critical but realistic.

AVAILABLE TOOLS for additional tasks (use ONLY these):
- search_documents: Search company reports and handbooks
- query_database: SQL on PostgreSQL
- calculate: Math with numbers only

ACTUAL DATABASE SCHEMA (only this data exists):
{schema}

IMPORTANT RULES:
- Do NOT suggest tasks that require data not in the schema above
- Do NOT invent tools — only search_documents, query_database, calculate
- If data doesn't exist in our sources, acknowledge the gap but don't suggest impossible tasks
- Score 7+ if the report is useful despite minor gaps

Respond with JSON:
{{
    "quality_score": 1-10,
    "is_complete": true/false,
    "gaps": ["what's missing"],
    "additional_tasks": [{{"id": 99, "description": "...", "tool": "search_documents or query_database or calculate", "depends_on": []}}]
}}"""
            },
            {
                "role": "user",
                "content": f"GOAL: {goal}\n\nCOMPLETED WORK:\n{work_summary}",
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

def autonomous_rag(goal:str,max_iterations:int=3)->str:

    print(f"\n Planning...")
    tasks=plan(goal)

    for task in tasks:
        deps=task.get("depends_on",[])
        deps_str=f" (after {deps})" if deps else ""
        print(f"  Task {task['id']}:[{task.get('tool','synthesize')}] {task['description'][:70]}{deps_str}")
    
    completed_results={}
    final_report=""

    for iteration in range(max_iterations):
        print(f"\n{'-'*60}")
        print(f"ITERATION {iteration+1}")
        print(f"\n{'-'*60}")

        print(f"\n EXECUTING...")

        progress=True
        while progress:
            progress=False
            for task in tasks:
                task_key=f"task_{task["id"]}"
                if task_key in completed_results:
                    continue

                deps=task.get("depends_on",[])
                deps_met=all(f"task_{d}" in completed_results for d in deps)
                if not deps_met:
                    continue

                progress=True
                tool=task.get("tool","synthesize")
                print(f"\n Task {task['id']}:{task['description'][:60]}...")

                result=execute_task(task,completed_results)
                completed_results[task_key]=result

                preview=result[:150].replace("\n"," ")
                print(f"  ->{preview}...")

                if tool=="synthesize":
                    final_report=result
        
        print(f"\n🔍 EVALUATING...")
        evaluation = evaluate(goal, completed_results)

        score = evaluation.get("quality_score", 0)
        is_complete = evaluation.get("is_complete", False)
        gaps = evaluation.get("gaps", [])
        new_tasks = evaluation.get("additional_tasks", [])

        print(f"   Score: {score}/10")
        print(f"   Complete: {is_complete}")
        for gap in gaps:
            print(f"   Gap: {gap}")

        # Phase 4: Deliver or Refine
        if score >= 7:
            print(f"\n✅ Quality threshold met! Delivering.")
            return final_report

        if not new_tasks:
            print(f"\n⚠️  Below threshold but no new tasks. Delivering best effort.")
            return final_report

        # Add new tasks for refinement
        print(f"\n🔧 REFINING — adding {len(new_tasks)} tasks...")
        max_id = max(t["id"] for t in tasks)

        for i, new_task in enumerate(new_tasks):
            new_id = max_id + i + 1
            new_task["id"] = new_id
            if "depends_on" not in new_task:
                new_task["depends_on"] = []
            tasks.append(new_task)
            print(f"   New Task {new_id}: [{new_task.get('tool', 'search_documents')}] {new_task['description'][:60]}")

        # Add re-synthesis
        synth_id = max_id + len(new_tasks) + 1
        all_data_ids = [t["id"] for t in tasks if t.get("tool") != "synthesize"]
        tasks.append({
            "id": synth_id,
            "description": "Re-synthesize all findings into an improved report",
            "tool": "synthesize",
            "depends_on": all_data_ids,
        })
        print(f"   New Task {synth_id}: [synthesize] Re-synthesize everything")

    print(f"\n⚠️  Max iterations reached.")
    return final_report


# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Verify connections
    stats = index.describe_index_stats()
    print(f"Pinecone: {stats.total_vector_count} vectors")

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM employees")
        emp_count = cursor.fetchone()[0]
        conn.close()
        print(f"PostgreSQL: {emp_count} employees")
    except Exception as e:
        print(f"PostgreSQL error: {e}")
        print("Run project 4's setup_data.py first.")
        exit()

    print(f"\n🤖 NovaTech Autonomous Agent (Pinecone + PostgreSQL)")
    print("   Give me a GOAL, not a question.")
    print("   Type 'quit' to exit\n")

    while True:
        try:
            goal = input("Goal: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not goal:
            continue
        if goal.lower() in ("quit", "exit", "q"):
            break

        report = autonomous_rag(goal)

        print(f"\n{'='*60}")
        print("📄 FINAL REPORT")
        print(f"{'='*60}")
        print(report)
        print()

