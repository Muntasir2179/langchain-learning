import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()

model = ChatOllama(model='llama2')

result = model.invoke("what is 81 divided by 9?")
print(result.content)
