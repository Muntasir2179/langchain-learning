import os
from rich import print
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


# setting up vector database directory
persistent_directory = os.path.join(os.path.dirname(os.path.abspath("__file__")), "chunking", "db", "recursive_chunk_db")

# loading the document
loader = PyMuPDFLoader(file_path="chunking/docs/The-Army-Regulations.pdf")
data = loader.load_and_split(
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=300
    )
)

# loading the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1",
                                        cache_folder="embedding_model",
                                        model_kwargs={"device": 'cuda:0'})

# Creating or loading vector store with the embedding function
if os.path.exists(persistent_directory):
    db = Chroma(embedding_function=embedding_model, 
                persist_directory=persistent_directory)
else:
    db = Chroma.from_documents(documents=data,
                               embedding=embedding_model,
                               persist_directory=persistent_directory)
    
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    result = db.search(query=query, search_type="similarity")
    print(result)
    print()

