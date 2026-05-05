import json
from config.settings import client,LLM_MODEL

system_prompt="""
You are a query router for an e-commerce analytics system.

Available agents:
-sales:Order and revenue questions (order counts,revenue by period/region/product, payment methods, order status)
-customer:Customer and review questions (customer segments, satisfaction, review scores, complaints, customer locations)
-operations:Seller and delivery questions (seller performance, shipping times, delivery status, logistics)
-research: Strategy and industry questions (technology trends, market analysis, AI/digital transformation - from McKinsey report)

RULES:
-Simple single-domain question -> ONE agent
-Cross-domation question -> MULTIPLE agents with sub-questions
-Business reviews or summary -> ALL relevant agents
-Each sub-question must be specific and answerable by that agent alone

Respond with JSON:
{
"agents":["sales","customer"],
"reasoning":"why these agents",
"sub_questions":{"sales":"specific question","customer":"specific question"}
}
"""

def route(question):
    """clarify questions -> assign to specialist agent(s).
    Returns: {"agents":[...],"reasoning":"...","sub_questions":{...}}"""

    response=client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":question}
        ],
        temperature=0,
        response_format={"type":"json_object"},
    )
    return json.loads(response.choices[0].message.content)