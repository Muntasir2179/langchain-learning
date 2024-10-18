import os
from rich import print
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# setting up vector database directory
persistent_directory = os.path.join(os.path.dirname(os.path.abspath("__file__")), "chunking", "db", "page_wise_chunk_db")

# loading the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-base-en-v1.5",
                                        cache_folder="embedding_model",
                                        model_kwargs={"device": 'cuda:0',
                                                      "trust_remote_code":True},
                                        encode_kwargs={'batch_size':1})

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    result = db.search(query=query, search_type="similarity")
    print(result)

