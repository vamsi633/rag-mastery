import json
from pydantic import BaseModel,Field
from ddgs import DDGS

class SearchInput(BaseModel):
    query:str=Field(description="the search query for the live web")
class CalculatorInput(BaseModel):
    expression:str=Field(description="Math expression, e.g.,'math.sqrt(16*12)'.")


def web_search(query: str) -> str:
    """Searches the web for current facts."""
    try:
        with DDGS() as ddgs:
            # Add a timeout and region to help bypass simple blocks
            results = [r for r in ddgs.text(query, max_results=3)]
            
            if not results:
                # Tell the agent explicitly that the search engine blocked us
                return "Error: Search engine returned no results. Try a different query or tell the user the service is down."
            
            return str(results)
    except Exception as e:
        return f"System Error in web_search: {str(e)}"

def calculate(expression:str)->str:
    """Evaluate a math expression safely"""
    import math
    try:
        allowed={"math":math,"sqrt":math.sqrt,"pow":math.pow}
        return str(eval(expression,{"__bultins__":{}},allowed))
    except Exception as e:
        return f" Calc error: {e}"

Tool_REGISTRY={
    "web_search":web_search,
    "calculate":calculate
}
