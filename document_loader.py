import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_document(file_path: str):
    """
    Loads a PDF document, extracts text, and chunks it.
    
    Architectural decision: We use RecursiveCharacterTextSplitter with chunk overlap.
    Chunk overlap ensures that context isn't lost at the boundaries of chunks, 
    especially when an important paragraph or sentence spans across two chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load PDF using PyPDFLoader which automatically extracts page numbers into metadata
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    splits = text_splitter.split_documents(docs)
    return splits
