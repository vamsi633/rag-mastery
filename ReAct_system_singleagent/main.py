from agent import ReActAgent

if __name__=="__main__":
    agent=ReActAgent()

    query="What is the current stock price of nvidia and what is the result if we multiple stock price by 2 ?"
    print(f"User: {query}\n")

    result=agent.run(query)
    print(f"\nAgent: {result}")