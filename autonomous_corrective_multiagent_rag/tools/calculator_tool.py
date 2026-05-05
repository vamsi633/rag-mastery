def calculator(expression):
    try:
        allowed=set("0123456789.+-*/()")
        if not all(c in allowed for c in expression):
            return "Error : Only numbers and +,-,*,/ allowed"
        result=eval(expression,{"__builtins__":{}},{})
        return str(round(result,4))
    except Exception as e:
        return f"Calculation error: {e}"