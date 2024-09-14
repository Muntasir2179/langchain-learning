import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath("__file__"))
books_dir = os.path.join(current_dir, "rag", "books")
db_dir =  os.path.join(current_dir, "rag", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exists. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The file {books_dir} does not exists. Please check the path."
        )
    
    # list all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # read the text content from the file
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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
