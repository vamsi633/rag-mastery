"""
Project 6 — Step 1: Why Multi-Agent?

One agent with 10 tools vs specialized agents with 2-3 tools each.
Let's see the difference in prompt size and clarity.
"""

# ── Single agent system prompt (Project 4/5 style) ──

single_agent_prompt = """You are an AI Business Operations Assistant.

You have access to:
1. Company documents via search_documents
2. PostgreSQL database with employees, sales, tickets via query_database
3. Calculator for math
4. Memory of past conversations

Tables:
  employees (id, name, department, role, salary, hire_date, status, office)
  sales (id, deal_name, customer, amount, region, sales_rep, status, close_date, product)
  tickets (id, customer, issue, product, priority, status, assigned_to, created_date)

Rules:
- Call get_database_schema BEFORE writing SQL
- Search ONE topic at a time
- Use calculate for math
- Cite sources
- Use PostgreSQL syntax
- Double quotes for columns
- Handle HR questions about benefits, policies, compensation
- Handle sales questions about deals, pipeline, revenue
- Handle support questions about tickets, SLAs, customer issues
- Handle strategy questions about company direction, risks
- Cross-reference documents with database when needed"""

# ── Multi-agent prompts (Project 6 style) ──

hr_agent_prompt = """You are an HR specialist agent.
You ONLY handle employee-related questions.

Your tools:
- query_database: SQL on the employees table
  Columns: id, name, department, role, salary, hire_date, status, office
- search_documents: Search the employee handbook

You know about: headcount, salaries, departments, hiring, turnover, 
benefits, PTO, health insurance, remote work policy."""

sales_agent_prompt = """You are a Sales analyst agent.
You ONLY handle sales and revenue questions.

Your tools:
- query_database: SQL on the sales table
  Columns: id, deal_name, customer, amount, region, sales_rep, status, close_date, product
- calculate: Math for revenue totals, growth rates

You know about: deals, pipeline, revenue by region, win rates, 
deal sizes, sales rep performance."""

support_agent_prompt = """You are a Customer Support analyst agent.
You ONLY handle support ticket questions.

Your tools:
- query_database: SQL on the tickets table
  Columns: id, customer, issue, product, priority, status, assigned_to, created_date
  
You know about: open tickets, resolution times, SLAs, 
customer issues, priority distribution."""

# ── Compare ──

print("SINGLE AGENT vs MULTI-AGENT")
print("=" * 50)
print(f"\nSingle agent prompt: {len(single_agent_prompt)} characters")
print(f"  Has to know EVERYTHING")
print(f"  Confused about which table to use")
print(f"  Long prompt = more token cost per query")

print(f"\nSpecialist agent prompts:")
print(f"  HR Agent:      {len(hr_agent_prompt)} chars — knows ONLY employees")
print(f"  Sales Agent:   {len(sales_agent_prompt)} chars — knows ONLY deals")
print(f"  Support Agent: {len(support_agent_prompt)} chars — knows ONLY tickets")

print(f"""

WHY THIS IS BETTER:

1. FOCUSED: Each agent knows only its domain
   HR agent never sees the sales table → can't write wrong SQL
   Sales agent never sees tickets → no confusion

2. CHEAPER: Smaller prompts = fewer tokens per call
   Single agent: {len(single_agent_prompt)} chars EVERY query
   Specialist:   ~400 chars per query (only the relevant agent)

3. BETTER QUALITY: Specialists outperform generalists
   An HR agent with 3 tools beats a generalist with 10 tools
   because the LLM has less to juggle

4. SCALABLE: Adding a new domain = adding a new agent
   Want to add a Finance agent? Create it separately.
   Don't touch the existing agents at all.

5. DEBUGGABLE: When something goes wrong, you know WHICH agent failed
   "The Sales agent wrote bad SQL" vs "The agent wrote bad SQL"
""")