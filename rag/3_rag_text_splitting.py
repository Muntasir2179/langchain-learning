import os
from typing import List
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# defining the directory containing the text file
current_dir = os.path.dirname(os.path.abspath("__file__"))
file_path = os.path.join(current_dir, "rag", "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "rag", "db")

# check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exists. Please check path."
    )

# read the text content from the file
loader = TextLoader(file_path)
documents = loader.load()

# define the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                        model_kwargs={'device': 'cuda'},
                                        encode_kwargs={'normalize_embeddings': False})


# function to create and persist vector store
def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n----- Creating Vector Store {store_name} -----")
        db = Chroma.from_documents(documents=docs,
                                   embedding=embedding_model,
                                   persist_directory=persistent_directory)
        print(f"\n----- Finished creating vector store {store_name} -----")
    else:
        print(f"Vector store {store_name} already exists. NO need to initialize.")


# 1. Character-based splitting
# Splits text into chunks based on a specified number of characters.
# Useful for consistent chunk sizes regardless of content structure.
print("\n----- Using Character-based Splitting -----")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")

# 2. Sentence-based splitting
# Splits text into chunks based on sentences, ensuring chunks end at sentence boundaries.
# Ideal for maintaining semantic coherence within chunks.
print("\n----- Using Sentence-based Splitting -----")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_tokens = sent_splitter.split_documents(documents)
create_vector_store(sent_tokens, "chroma_db_sent")

# 3. Token-based splitting
# Splits text into chunks based on tokens (word or subwords), using tokenizers like GPT-2
# Useful for transformer models with strict token limits
print(f"\n----- Using Token-based Splitting -----")
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")

# 4. Recursive CHaracter-based Splitting
# Attempt to split text at natural boundaries (sentences, paragraphs) within character limit.
# Balances between maintaining coherence and adhering to character limits.
print("\n----- Using Recursive Character-based Splitting -----")
rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, "chroma_db_rec_char")

# 5. Custom Splitting
# Allows creating custom splitting logics based on specific requirements
# Useful for documents with unique structure that standard splitters can't handle.
print(f"\n----- Using Custom Splitting -----")

class CustomTextSplitter(TextSplitter):
    def split_text(self, text: str):
        # custom logic to splitting text
        return text.split("\n\n")   # Example: split by paragraph


custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")


# function to query a vector store
def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n----- Querying the Vector Store {store_name} -----")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1}
        )
        relevant_docs = retriever.invoke(query)

        # display the relevant results with metadata
        print(f"\n----- Relevant Documents for {store_name} -----")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exists.")


query = "How did Juliet die?"

# query each vector store
query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
query_vector_store("chroma_db_custom", query)
