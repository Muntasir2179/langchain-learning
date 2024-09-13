from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser
import os


load_dotenv()


model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.environ["GROQ_API_KEY"])

positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    (
        "human",
        "Generate a thank you note for this positive feedback: {feedback}."
    )
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    (
        "human",
        "Generate a response addressing this negative feedback: {feedback}."
    )
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    (
        "human",
        "Generate a request for more details for this neutral feedback: {feedback}."
    )
])

escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    (
        "human",
        "Generate a message to escalate this feedback to a human agent: {feedback}."
    )
])

# defining the feedback classification template
classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    (
        "human",
        "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."
    )
])

# defining the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()   # positive feedback
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()   # negative feedback
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()    # neutral feedback
    ),
    escalate_feedback_template | model | StrOutputParser()   # default branch will run when all other will not run
)

# create the classification chain
classification_chain = classification_template | model | StrOutputParser()

# combine classification and response generation into one chain
chain = classification_chain | branches

# run the chain
# Good review: "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review: "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review: "The product is okay. It works as expected but nothing exceptional."
# Default: "I am not sure about the product yet. Can you tell me more about it's features and benefits."

review = "I am not sure about the product yet. Can you tell me more about it's features and benefits."
result = chain.invoke({"feedback": review})
print(result)
