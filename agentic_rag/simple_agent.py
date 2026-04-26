from dotenv import load_dotenv
from openai import OpenAI
import json
import os
from datetime import datetime
load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def calculate(expression):
    """The actual calculator. This runs on your computer not the llm"""

    try:
        return str(eval(expression))
    except:
        return "Error"
    
def get_current_time(timezone:str="UTC")->str:
    """Get the current time."""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def lookup_employee(name:str)->str:
    """Fake employee lookup — we'll replace with real DB later."""
    fake_data = {
        "sarah": "Sarah Chen | Engineering | VP Engineering | $195,000",
        "marcus": "Marcus Johnson | Engineering | Senior Engineer | $165,000",
        "robert": "Robert Kim | Sales | VP Sales | $185,000",
    }
    result = fake_data.get(name.lower().split()[0], None)
    return result if result else f"No employee found matching '{name}'"

TOOL_FUNCTIONS={
    "calculate":calculate,
    "get_current_time":get_current_time,
    "lookup_employee":lookup_employee,
}

TOOL_DEFINITIONS=[
    {
        "type":"function",
        "function":{
            "name":"calculate",
            "description":"Do Math. Use this for any calculation",
            "parameters":{
                "type":"object",
                "properties":{
                    "expression":{
                        "type":"string",
                        "description":"Math expression like '15*4+10"
                    }
                },
                "required":["expression"],
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"get_current_time",
            "description":"Get the current date and time",
            "parameter":{
                "type":"object",
                "properties":{
                    "timezone":{"type":"string","description":"TimeZone (default UTC)"}
                },
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"lookup_employee",
            "description":"Look up employee info by name. returns their department,role, and salary",
            "parameters":{
                "type":"object",
                "properties":{
                    "name":{"type":"string","description":"Employee name to serach for"},
                },
                "required":["name"],
            },
        },
    },
]


def agent(question,max_steps=5):
    print(f"\n {question}")

    messages=[
        {"role":"system","content":"You are a helpful assistant. use your tools when needed"},
        {"role":"user","content":question}
    ]

    for step in range(max_steps):
        response=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            temperature=0,
        )

        message=response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name=tool_call.function.name
                args=json.loads(tool_call.function.arguments)

                print(f"  Step {step+1}: LLM calls {func_name}({args})")

                func=TOOL_FUNCTIONS.get(func_name)
                if func:
                    result=func(**args)
                else:
                    result=f"Unknown tool: {func_name}"
                print(f"    ->{result}")
                messages.append({"role":"tool","tool_call_id":tool_call.id,"content":str(result)}),
        else:
            print(f"done in {step+1} steps")
            return message.content
    return "max steps reached"
# Uses calculator
answer = agent("What is 1247 * 83?")
print(f"💬 {answer}\n")

# Uses employee lookup
answer = agent("What department does Sarah work in and what is her salary?")
print(f"💬 {answer}\n")

# Uses MULTIPLE tools in one question
answer = agent("Look up Robert's salary and calculate how much a 15% raise would be.")
print(f"💬 {answer}\n")

# Uses time tool
answer = agent("What time is it right now?")
print(f"💬 {answer}\n")

# No tools needed
answer = agent("What is the capital of Japan?")
print(f"💬 {answer}\n")