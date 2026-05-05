"""
core/synthesizer.py — Combine results from multiple agents.
"""

from config.settings import client, LLM_MODEL


def synthesize(question, agent_results):
    """Combine findings from multiple specialist agents into one answer."""
    results_text = ""
    for agent_name, result in agent_results.items():
        results_text += f"\n--- {agent_name.upper()} AGENT ---\n{result}\n"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Combine findings from multiple agents into one clear, unified answer. Use specific numbers. Don't repeat data. Be concise and actionable."
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nFindings:\n{results_text}\n\nProvide a unified answer."
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content