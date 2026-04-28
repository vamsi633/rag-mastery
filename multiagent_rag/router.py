from dotenv import load_dotenv
from openai import OpenAI
import os
import json

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt="""
You are a query router. Classify the question and assign it to the right agent(s)

Avaiable agents:
-hr:Employee questions (headcount,salaries,departments,benefits,PTO, hiring,turnover,remote work policy)
--sales:Revenue and deal questions (deals,pepeline,revenue by region, products,win rates,customers)
- support: Ticket and customer issue questions (open tickets, resolution, SLAs, customer complaints, priorities)
- research: Strategy and document questions (Q3 report, company strategy, risk factors, market analysis, future plans)

RULES:
-Simple questions->ONE agent
-Complex questions->MULTIPLE agents,each with its own sub-question
-if the question spans two domains,assign BOTH agents
-Each sub-question should be specific and answerable by that agent alone

Respond WITH JSON:
{
"agents":["hr],
"reasoning":"This is about employee data",
"sub_questions":{"hr":"The specific question for HR agent"}
}

For multi_agent:
{
"agents":["hr","sales"],
"reasoning":"Needs both team data and revenue data",
"sub_questions":{
"hr":"What is the headcount by department?",
"sales":"What is the total revenue by region?"
}
}
"""
def route(question:str)->dict:

    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":question},

        ],
        temperature=0,
        response_format={"type":"json_object"},
    )

    return json.loads(response.choices[0].message.content)

if __name__=="__main__":
    questions = [
        # Single agent — clear routing
        "How many employees are in engineering?",
        "What is our total sales pipeline value?",
        "How many critical tickets are open?",
        "What does the Q3 report say about risks?",
        
        # Multi-agent — needs multiple specialists
        "Is our support team large enough to handle the ticket volume?",
        "Which region has the best revenue per employee?",
        "Prepare a complete business health summary",
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"❓ {q}")
        result = route(q)
        agents = result.get("agents", [])
        reasoning = result.get("reasoning", "")
        sub_q = result.get("sub_questions", {})

        print(f"   Agents: {agents}")
        print(f"   Reason: {reasoning}")
        for agent, question in sub_q.items():
            print(f"   → {agent}: {question}")