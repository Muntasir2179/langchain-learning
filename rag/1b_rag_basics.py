import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


current_dir = os.path.dirname(os.path.abspath("__file__"))
persistent_directory = os.path.join(current_dir, "rag", "db", "chroma_db")

# loading the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                        model_kwargs={'device': 'cuda'},
                                        encode_kwargs={'normalize_embeddings': False})

# loading the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embedding_model)

query = "Who is Odysseus' wife?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4}
)

relevant_docs = retriever.invoke(query)

print(len(relevant_docs))

# display the relevant results with metadata
print("\n---- Relevant Documents ----\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
