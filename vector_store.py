from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from config import VECTOR_DB_DIR

def get_embeddings():
    """
    Initializes the embedding model using FastEmbed.
    This avoids PyTorch entirely and uses ONNX, bypassing Windows Defender Application Control DLL blocks.
    """
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def create_vector_store(documents):
    """
    Creates and persists a ChromaDB vector store from document chunks.
    """
    embeddings = get_embeddings()
    
    # Generate deterministic IDs so that re-indexing the same document overwrites instead of duplicates
    import hashlib
    ids = [hashlib.md5(doc.page_content.encode('utf-8')).hexdigest() for doc in documents]
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        persist_directory=VECTOR_DB_DIR
    )
    return vectorstore

def get_vector_store():
    """
    Loads the persistent ChromaDB vector store.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR, 
        embedding_function=embeddings
    )
    return vectorstore

def get_retriever():
    """
    Returns a retriever interface from the vector store.
    We retrieve the top 4 most relevant chunks to provide enough context.
    """
    vectorstore = get_vector_store()
    return vectorstore.as_retriever(search_kwargs={"k": 4})
