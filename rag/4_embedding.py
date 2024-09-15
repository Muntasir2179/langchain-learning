import os
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma


current_dir = os.path.dirname(os.path.abspath("__file__"))
file_path = os.path.join(current_dir, "rag", "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "rag", "db")


# check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exists. Please check path."
    )

# read the text content from the file
loader = TextLoader(file_path)
documents = loader.load()

# split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# display information about the split documents
print(f"\n----- Document Chunks Information -----")
print(f"Number of document chunks: {len(docs)}\n")
print(f"Sample chunk:\n{docs[0].page_content}\n")

# function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n----- Creating vector store {store_name} -----")
        Chroma.from_documents(documents=docs,
                              embedding=embeddings,
                              persist_directory=persistent_directory)
        print(f"----- Finished creating vector store {store_name} -----")
    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")


# 1. Ollama Embeddings
# Uses Ollama's embedding models.
print("\n--- Using Ollama Embeddings ---")
ollama_embeddings = OllamaEmbeddings(model="bge-m3:latest")
create_vector_store(docs, ollama_embeddings, "chroma_db_ollama")

# 2. Hugging Face Transformers
# Uses models from the Hugging Face library.
# Ideal for leveraging a wide variety of models for different tasks.
# Note: Running Hugging Face models locally on your machine incurs no direct cost other than using your computational resources.
# Note: Find other models at https://huggingface.co/models?other=embeddings
print("\n--- Using Hugging Face Transformers ---")
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")

print("Embedding demonstrations for Ollama and HuggingFace completed.")
