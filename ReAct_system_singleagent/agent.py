from openai import OpenAI
import json
from tools import Tool_REGISTRY
import os
from dotenv import load_dotenv

load_dotenv()

class ReActAgent:
    def __init__(self,model="gpt-4o-mini",max_iters=5):
        self.client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model=model
        self.max_iters=max_iters
        self.history = [
    {
        "role": "system", 
        "content": """You are a precise research agent. 
        Step 1: Search for the current price. 
        Step 2: Extract the NUMERIC value from the search results.
        Step 3: Pass ONLY the number and math symbols to the calculate tool. 
        
        CRITICAL: Never use words or variables like 'sqrt(price)' in the calculator. 
        Always use the actual number, like 'math.sqrt(209.25)'."""
    }
]

    def get_tool_schemas(self):
        # Professional schema for OpenAI
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform math.",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"]
                    }
                }
            }
        ]
    
    def run(self,user_query):
        self.history.append({"role":"user","content":user_query})

        for i in range(self.max_iters):
            response=self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=self.get_tool_schemas()
            )

            message=response.choices[0].message
            self.history.append(message)


            if not message.tool_calls:
                return message.content
            
            for tool_call in message.tool_calls:
                name=tool_call.function.name
                args=json.loads(tool_call.function.arguments)

                print(f"Executing Tool: {name} with {args}")
                result=Tool_REGISTRY[name](**args)

                self.history.append({
                    "role":"tool",
                    "tool_call_id":tool_call.id,
                    "content":result
                })
        return "Max iterations reached."
