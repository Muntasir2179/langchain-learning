from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os


load_dotenv()


model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.environ["GROQ_API_KEY"])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian who tells joke about {topic}."),
    ("human", "Tell me {joke_count} jokes.")
])

# Creating the combined chain using LangChain expression language
chain = prompt_template | model | StrOutputParser()

# run the chain
result = chain.invoke({"topic": "Lawyers", "joke_count": 3})

# output
print(result)