from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import psycopg2
import json
import os


load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.Index(os.getenv("PINECONE_INDEX"))
DATABASE_URL=os.getenv("DATABASE_URL")


def run_sql(sql: str) -> str:
    """Execute a read-only SQL query."""
    if not sql.upper().strip().startswith("SELECT"):
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
        result = ""
        for row in rows[:15]:
            result += str(dict(zip(columns, row))) + "\n"
        return result
    except Exception as e:
        return f"SQL Error: {e}"


def search_docs(query: str) -> str:
    """Search documents in Pinecone."""
    query_emb = client.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding

    results = index.query(vector=query_emb, top_k=4, include_metadata=True)

    output = ""
    for match in results["matches"]:
        text = match["metadata"]["text"]
        source = match["metadata"].get("source", "?")
        page = match["metadata"].get("page", "?")
        score = round(match["score"], 3)
        output += f"[{source} p{page}, score:{score}] {text}\n\n"
    return output if output else "No results."


def calc(expression: str) -> str:
    """Safe math calculator."""
    try:
        allowed = set("0123456789.+-*/() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        return str(round(eval(expression, {"__builtins__": {}}, {}), 2))
    except Exception as e:
        return f"Error: {e}"

AGENTS={
    "hr": {
        "system_prompt": """You are an HR specialist. Answer ONLY employee-related questions.

Your tools:
- query_database: SQL on the employees table
  Columns: id(int), name(text), department(text), role(text), salary(numeric), hire_date(date), status(text), office(text)
  Distinct departments: Engineering, Sales, Customer Success, Product, Marketing, HR
  Distinct statuses: active, left
  Distinct offices: San Francisco, New York, London, Tokyo, Bangalore
- search_documents: Search the employee handbook for policies

Use double quotes around column names. PostgreSQL syntax only.
Always provide specific numbers. Be concise.""",

        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "SQL on employees table. Use for headcount, salaries, departments, turnover.",
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search employee handbook. Use for policies, benefits, PTO, remote work.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ],
        "tool_map": {
            "query_database": lambda sql, **kw: run_sql(sql),
            "search_documents": lambda query, **kw: search_docs(query),
        },
    },

    "sales": {
        "system_prompt": """You are a Sales analyst. Answer ONLY sales and revenue questions.

Your tools:
- query_database: SQL on the sales table
  Columns: id(int), deal_name(text), customer(text), amount(numeric), region(text), sales_rep(text), status(text), close_date(date), product(text)
  Distinct regions: North America, Europe, Asia Pacific, Latin America
  Distinct statuses: closed_won, negotiating, proposal, lost
  Distinct products: CloudSync Enterprise, DevTools Pro, NovaTech Analytics, Mobile SDK, CloudSync Starter
- calculate: Math for totals, percentages, averages

Use double quotes around column names. PostgreSQL syntax only.
Always provide specific numbers. Be concise.""",

        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "SQL on sales table. Use for deals, revenue, pipeline, win rates.",
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Math calculations. Use for totals, percentages.",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            },
        ],
        "tool_map": {
            "query_database": lambda sql, **kw: run_sql(sql),
            "calculate": lambda expression, **kw: calc(expression),
        },
    },

    "support": {
        "system_prompt": """You are a Customer Support analyst. Answer ONLY support ticket questions.

Your tools:
- query_database: SQL on the tickets table
  Columns: id(int), customer(text), issue(text), product(text), priority(text), status(text), assigned_to(text), created_date(date), resolved_date(date)
  Distinct priorities: critical, high, medium, low
  Distinct statuses: resolved, in_progress, open

Use double quotes around column names. PostgreSQL syntax only.
Always provide specific numbers. Be concise.""",

        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "SQL on tickets table. Use for ticket counts, priorities, resolution status.",
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                },
            },
        ],
        "tool_map": {
            "query_database": lambda sql, **kw: run_sql(sql),
        },
    },

    "research": {
        "system_prompt": """You are a Research analyst. Answer questions using company documents.

Your tools:
- search_documents: Search company reports, handbooks, and policies

You know about: Q3 financial results, company strategy, risk factors,
market analysis, Q4 plans, growth trends.

Search for ONE topic at a time. Cite source and page number.
Be concise and factual.""",

        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search company documents. Use for strategy, risks, financial reports.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ],
        "tool_map": {
            "search_documents": lambda query, **kw: search_docs(query),
        },
    },
}

def run_specialist(agent_name:str,question:str,max_steps:int=5)->str:

    agent_config=AGENTS[agent_name]

    messages=[
        {"role":"system","content":agent_config["system_prompt"]},
        {"role":"user","content":question},
    ]


    tool_map=agent_config["tool_map"]
    tools=agent_config["tools"]

    print(f"\n  [{agent_name.upper()} AGENT] Processing: {question[:60]}")

    for step in range(max_steps):
        response=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            temperature=0,
        )

        message=response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name=tool_call.function.name
                args=json.loads(tool_call.function.arguments)

                print(f"  Step {step+1}: {func_name}")
                for k,v in args.items():
                    preview=str(v)[:80]
                    print(f"    {k}:{preview}")
                func=tool_map.get(func_name)
                if func:
                    try:
                        result=func(**args)
                    except TypeError:
                        result=func()
                else:
                    result=f"Unknown tool: {func_name}"

                preview=result[:120].replace("\n"," ")
                print(f"       ->{preview}...")

                messages.append({
                    "role":"tool",
                    "tool_call_id":tool_call.id,
                    "content":str(result),
                })
        else:
            print(f"  Done in {step+1} steps")
            return message.content
    return "Agent reached max steps."

def route(question: str) -> dict:
    """Classify question and assign to specialist agent(s)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a query router. Assign the question to the right agent(s).

Available agents:
- hr: Employee questions (headcount, salaries, departments, benefits, PTO, hiring, turnover, remote work)
- sales: Revenue and deal questions (deals, pipeline, revenue, products, win rates, customers)
- support: Ticket questions (open tickets, resolution, SLAs, customer complaints, priorities)
- research: Strategy and document questions (Q3 report, strategy, risks, market analysis, future plans)

For simple questions → ONE agent.
For complex/cross-domain questions → MULTIPLE agents, each with a sub-question.
For business reviews or summaries → ALL relevant agents.

Respond with JSON:
{
    "agents": ["hr", "sales"],
    "reasoning": "why",
    "sub_questions": {"hr": "specific question for HR", "sales": "specific question for Sales"}
}"""
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

def synthesize(question:str,agent_results:dict)->str:
    results_text=""
    for agent_name,result in agent_results.items():
        results_text+=f"\n--- {agent_name.upper()} AGENT FINDINGS ---\n{result}\n"
    
    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You combine findings from multiple agents into one clear, unified answer. Use specific numbers. Don't repeat the same data twice. Be concise and actionable."},
            {"role":"user","content":f"Question: {question}\n\nAgent findings:\n{results_text}\n\nProvide a unified answer"}
        ],
        temperature=0
    )
    return response.choices[0].message.content
def multi_agent_query(question: str) -> str:
    """
    The full pipeline:
    1. Router classifies the question
    2. Relevant specialist agents run
    3. If multiple agents → synthesizer combines results
    4. Return final answer
    """
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}")

    # Step 1: Route
    routing = route(question)
    agents = routing.get("agents", ["research"])
    reasoning = routing.get("reasoning", "")
    sub_questions = routing.get("sub_questions", {})

    print(f"\n  📡 ROUTER: {agents}")
    print(f"     Reason: {reasoning}")


    agent_results = {}
    for agent_name in agents:
        sub_q = sub_questions.get(agent_name, question)
        result = run_specialist(agent_name, sub_q)
        agent_results[agent_name] = result

    # Step 3: Synthesize if multiple agents
    if len(agent_results) == 1:
        final = list(agent_results.values())[0]
    else:
        print(f"\n  📝 SYNTHESIZER: Combining {len(agent_results)} agent results...")
        final = synthesize(question, agent_results)

    return final


if __name__ == "__main__":
    # Verify connections
    stats = index.describe_index_stats()
    print(f"Pinecone: {stats.total_vector_count} vectors")

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM employees")
        print(f"PostgreSQL: {cursor.fetchone()[0]} employees")
        conn.close()
    except:
        print("PostgreSQL connection failed. Run project4/setup_data.py first.")
        exit()

    print(f"\n🤖 NovaTech Multi-Agent System")
    print("   Specialists: HR | Sales | Support | Research")
    print("   Type 'quit' to exit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        answer = multi_agent_query(question)
        print(f"\n💬 {answer}\n")







    

