"""
tools/memory_tool.py — Conversation memory using PostgreSQL.
"""

import psycopg2
from config.settings import DATABASE_URL


def save(query, answer, agent_used, tools_called):
    """Save an interaction to conversation memory."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO conversation_memory (query, answer, agent_used, tools_called)
               VALUES (%s, %s, %s, %s)""",
            (query, answer[:2000], agent_used, tools_called),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  [Memory] Save error: {e}")


def recall(limit=5):
    """Retrieve recent conversations."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT query, answer, agent_used, timestamp
               FROM conversation_memory
               ORDER BY timestamp DESC LIMIT %s""",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No previous conversations."

        output = ""
        for q, a, agent, ts in rows:
            output += f"[{ts}] ({agent}) Q: {q}\nA: {a[:200]}...\n\n"
        return output

    except Exception as e:
        return f"Memory error: {e}"