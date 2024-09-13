from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
import os


load_dotenv()


model = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.environ["GROQ_API_KEY"])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are expert product reviewer."),
    ("human", "List the main features of the product {product_name}.")
])


# define pros analysis steps
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are expert product reviewer."),
            (
                "human", 
                "Given the features: {features}, list the pros of these features."
            )
        ]
    )
    return pros_template.format_prompt(features=features)


# define cons analysis steps
def analysis_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are expert product reviewer."),
            (
                "human", 
                "Given the features: {features}, list the cons of these features."
            )
        ]
    )
    return cons_template.format_prompt(features=features)


# combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analysis_cons(x)) | model | StrOutputParser()
)


# pros and cons will be parallelly computed as we defined separate chains for them
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# run the chain
result = chain.invoke({"product_name": "MacBook Pro"})
print(result)
