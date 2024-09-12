from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


model = ChatOllama(model="llama2")

chat_history = []


system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)   # Add system message to the chat history

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query))   # add user message

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))   # add AI message
    print(f"AI: {response}")

print("----- Message History -----")
print(chat_history)
