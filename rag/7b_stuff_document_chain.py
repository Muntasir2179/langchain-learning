import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath("__file__"))
persistent_directory = os.path.join(current_dir, "rag", "db", "chroma_db_with_metadata")

# loading the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                        model_kwargs={'device': 'cuda'},
                                        encode_kwargs={'normalize_embeddings': False})

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatOpenAI model
llm = ChatGroq(model="mixtral-8x7b-32768")


qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# creating a dummy context document
#### Important ####
# The stuff_document_chain does all the heavy lifting of retrieved context parsing. Whether it is one ore more
# retrieved contexts, it will combine them all to formulate a single context
context = [Document(page_content='''To learn more about LangChain and its wide range of applications, be sure to check out 
the comprehensive documentation and tutorials available on the official website:
LangChain Documentation: https://python.langchain.com/''')]

print("\n------- Response -------\n")
print(question_answer_chain.invoke({"input": "How can I learn about LangChain", "chat_history": [], "context": context}))
