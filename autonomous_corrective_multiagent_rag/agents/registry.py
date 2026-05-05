"""
agents/registry.py — All specialist agent configurations.
Add a new agent = add a new entry here. Nothing else changes.
"""

from tools.sql_tool import execute_query, get_schema
from tools.search_tool import search, search_raw
from tools.calculator_tool import calculator
from agents.base_agent import run


# ── Tool definitions (OpenAI format) ──

SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Run a SELECT query on PostgreSQL.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "PostgreSQL SELECT query. Use double quotes for column names."}},
            "required": ["sql"],
        },
    },
}

SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "get_schema",
        "description": "Get database tables, columns, types, and sample data. ALWAYS call before writing SQL.",
        "parameters": {"type": "object", "properties": {}},
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the McKinsey Technology Trends 2024 report.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Focused search query"}},
            "required": ["query"],
        },
    },
}

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Math. Only numbers and +, -, *, /.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


# ── Shared tool map ──

TOOL_MAP = {
    "query_database": lambda sql, **kw: execute_query(sql),
    "get_schema": lambda **kw: get_schema(),
    "search_documents": lambda query, **kw: search(query),
    "calculate": lambda expression, **kw: calculator(expression),
}


# ── Agent configurations ──

AGENTS = {
    "sales": {
        "system_prompt": """You are a Sales & Revenue analyst for a Brazilian e-commerce platform.
This dataset contains orders from 2016-2018. Do NOT filter by recent dates. Query ALL data unless asked for a specific period.
You answer questions about orders, revenue, payments, and products.


RULES:
- ALWAYS call get_schema before writing SQL
- Key tables: olist_orders_dataset, olist_order_items_dataset, olist_order_payments_dataset, olist_products_dataset, product_category_name_translation
- Use double quotes for column names
- Use PostgreSQL syntax
- Use calculate tool for math — don't do mental math
- Be concise, provide specific numbers
NOTE: Dataset contains orders from 2016-2018. Do NOT filter by recent dates unless specifically asked. Query ALL data by default.""",

        "tools": [SCHEMA_TOOL, SQL_TOOL, CALC_TOOL],
    },

    "customer": {
        "system_prompt": """You are a Customer Experience analyst for a Brazilian e-commerce platform.

You answer questions about customers, reviews, satisfaction, and complaints.
This dataset contains orders from 2016-2018. Do NOT filter by recent dates. Query ALL data unless asked for a specific period.
RULES:
- ALWAYS call get_schema before writing SQL
- Key tables: olist_customers_dataset, olist_order_reviews_dataset, olist_orders_dataset
- Use double quotes for column names
- Review scores are 1-5 (5 = best)
- Be concise, provide specific numbers
NOTE: Dataset contains reviews from 2016-2018. Do NOT filter by recent dates unless specifically asked. Query ALL data by default.""",

        "tools": [SCHEMA_TOOL, SQL_TOOL, CALC_TOOL],
    },

    "operations": {
        "system_prompt": """You are an Operations & Logistics analyst for a Brazilian e-commerce platform.

You answer questions about sellers, shipping, delivery times, and logistics.
This dataset contains orders from 2016-2018. Do NOT filter by recent dates. Query ALL data unless asked for a specific period.
RULES:
- ALWAYS call get_schema before writing SQL
- Key tables: olist_sellers_dataset, olist_orders_dataset, olist_order_items_dataset
- Delivery data: order_delivered_customer_date, order_estimated_delivery_date, order_purchase_timestamp
- Use double quotes for column names
- Be concise, provide specific numbers
NOTE: Dataset contains orders from 2016-2018. Do NOT filter by recent dates unless specifically asked. Query ALL data by default. seller_id is in olist_order_items_dataset, NOT in olist_orders_dataset — use a JOIN.""",

        "tools": [SCHEMA_TOOL, SQL_TOOL, CALC_TOOL],
    },

    "research": {
        "system_prompt": """You are a Research analyst with access to the McKinsey Technology Trends Outlook 2024 report.

You answer questions about technology trends, AI, digital transformation, industry analysis, and strategic insights.
This dataset contains orders from 2016-2018. Do NOT filter by recent dates. Query ALL data unless asked for a specific period.
RULES:
- Search for ONE topic at a time
- Cite source and page number
- Be concise and factual""",

        "tools": [SEARCH_TOOL],
    },
}


def run_agent(agent_name, question):
    """Run a specialist agent by name."""
    config = AGENTS[agent_name]
    return run(
        system_prompt=config["system_prompt"],
        tools=config["tools"],
        tool_map=TOOL_MAP,
        question=question,
        agent_name=agent_name,
    )


