from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
import os


load_dotenv()


model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.environ["GROQ_API_KEY"])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian who tells joke about {topic}."),
    ("human", "Tell me {joke_count} jokes.")
])

# creating individual runnables (step in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# create the runnable sequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# run the chain
response = chain.invoke({"topic": "Lawyers", "joke_count": 3})

print(response)