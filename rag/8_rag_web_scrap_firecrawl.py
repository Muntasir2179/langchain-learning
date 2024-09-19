import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader


load_dotenv()

current_dir = os.path.dirname(os.path.abspath("__file__"))
db_dir = os.path.join(current_dir, "rag", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_firecrawl")

# loading the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                        model_kwargs={'device': 'cuda'},
                                        encode_kwargs={'normalize_embeddings': False})

def create_vector_store():
    """Crawl the website, split the content, create embeddings, and persist the vector store"""
    api_key = os.environ["FIRECRAWL_API_KEY"]
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")

    print("Begin crawling the website....")
    loader = FireCrawlLoader(api_key=api_key, url="https://www.apple.com/", mode="scrape")
    docs = loader.load()
    print("Finished crawling the website")

    # converting metadata values to strings if they are lists
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))
    
    # split the crawled content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    print(f"\n------ Document Chunks Information ------")
    print(f"Number of document chunks: {len(split_docs)}")
    print(f"Sample chunk:\n{split_docs[0].page_content}\n")
    
    print(f"\n------ Creating vector store in {persistent_directory} ------")
    db = Chroma.from_documents(documents=split_docs,
                               embedding=embedding_model,
                               persist_directory=persistent_directory)
    print(f"------ Finished creating vector store in {persistent_directory} ------")


if not os.path.exists(persistent_directory):
    create_vector_store()
else:
    print(f"Vector store {persistent_directory} already exists. No need to initialize.")

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embedding_model)


def query_vector_store(query):
    """Query the vector store with specific question."""
    retriever = db.as_retriever(search_type="similarity",
                                search_kwargs={
                                    "k": 3
                                })
    
    relevant_docs = retriever.invoke(query)

    print("\n------ Relevant Documents ------")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


query = "Apple Intelligence?"

query_vector_store(query)