import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


# define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath("__file__"))
file_path = os.path.join(current_dir, "rag", "books", "langchain_demo.txt")
persistent_directory = os.path.join(current_dir, "rag", "db", "chroma_db_with_metadata")

# check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exists. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exists. Please check the path."
        )
    
    # read the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # display information about the split documents
    print("\n---- Document Chunks Information ----\n")
    print(f"Number of document chunks: {len(docs)}\n")
    print(f"Sample chunk: \n{docs[0].page_content}\n")

    # loading the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                            model_kwargs={'device': 'cuda'},
                                            encode_kwargs={'normalize_embeddings': False})

    # Create a vector store and persist it automatically
    print("\n---- Creating Vector Store ----")
    db = Chroma.from_documents(documents=docs,
                               embedding=embedding_model,
                               persist_directory=persistent_directory)
else:
    print("Vector store already exists. No need to initialize.")