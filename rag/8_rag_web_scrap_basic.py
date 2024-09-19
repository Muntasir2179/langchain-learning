import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


load_dotenv()

current_dir = os.path.dirname(os.path.abspath("__file__"))
db_dir = os.path.join(current_dir, "rag", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_apple")

urls = ["https://www.apple.com/"]

loader = WebBaseLoader(urls)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print("\n------ Document Chunks Information ------")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample Chunk:\n{docs[0].page_content}\n")


# loading the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                        model_kwargs={'device': 'cuda'},
                                        encode_kwargs={'normalize_embeddings': False})

if not os.path.exists(persistent_directory):
    print(f"\n------ Creating vector store in {persistent_directory} ------")
    db = Chroma.from_documents(documents=docs,
                               embedding=embedding_model,
                               persist_directory=persistent_directory)
    print(f"------ Finished creating vector store in {persistent_directory} ------")
else:
    print(f"Vector store {persistent_directory} already exists. No need to initialize.")
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embedding_model)

retriever = db.as_retriever(search_type="similarity",
                            search_kwargs={
                                "k": 3
                            })

query = "What new products are announced on Apple.com?"

relevant_docs = retriever.invoke(query)

print("\n------ Relevant Docs ------")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

