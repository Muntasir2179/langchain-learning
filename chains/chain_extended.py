from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
import os


load_dotenv()


model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.environ["GROQ_API_KEY"])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a comedian who tells joke about {topic}."),
    ("human", "Tell me {joke_count} jokes.")
])

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# In order to include some additional function, we need to make that a runnable. Only then we can include that with in the chain.
# each runnable in the chain passes its generated output to the following runnable
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"topic": "Lawyers", "joke_count": 3})
print(result)