from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os


# loading the environment variables
load_dotenv()

# loading the chat LLM
model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.environ["GROQ_API_KEY"])


# PART 1: Create a ChatPromptTemplate using a template string
# print("----- Prompt from Template -----")
# template = "Tell me a joke about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({"topic": "cats"})
# result = model.invoke(prompt)
# print(result.content)


# PART 2: Prompt with Multiple placeholders
# print("\n----- Prompt with Multiple Placeholder -----")
# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} short story about a {animal}.
# Assistant: """
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})

# result = model.invoke(prompt)
# print(result.content)


# PART 3: Prompt with multiple placeholder
print("\n----- Prompt with Multiple Placeholder -----\n")
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}.
Assistant: """
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})

result = model.invoke(prompt)
print(result.content)
