import os
from rich import print
import matplotlib.pyplot as plt
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader


# function for plotting
def plot_lengths(lengths, cutoff):
    plt.figure(figsize=(20, 6))
    plt.plot(lengths, marker='o')
    plt.axhline(y=cutoff, color='r', linestyle='--', label='y = 5')
    plt.title('Lengths of contexts')
    plt.xlabel("i'th context")
    plt.ylabel('Length')
    plt.show()

# setting up vector database directory
persistent_directory = os.path.join(os.path.dirname(os.path.abspath("__file__")), "chunking", "db", "page_wise_chunk_db")

# loading the document
loader = PyMuPDFLoader(file_path="chunking/docs/The-Army-Regulations.pdf")
pages = loader.load()

filtered_pages = [single_page for single_page in pages if len(single_page.page_content) > 150]

# loading the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-base-en-v1.5",
                                        cache_folder="embedding_model",
                                        model_kwargs={"device": 'cuda:0',
                                                      "trust_remote_code":True},
                                        encode_kwargs={'batch_size':1})

# Creating or loading vector store with the embedding function
if os.path.exists(persistent_directory):
    db = Chroma(embedding_function=embedding_model, 
                persist_directory=persistent_directory)
    print("Database already exists, loading the database")
else:
    db = Chroma.from_documents(documents=filtered_pages,
                               embedding=embedding_model,
                               persist_directory=persistent_directory)
    print("Creating the vector database")
