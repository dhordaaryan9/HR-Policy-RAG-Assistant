import os

# Configuration parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Using HuggingFace embeddings for local free embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB_DIR = "./chroma_db"
UPLOADS_DIR = "./uploads"

# Ensure uploads directory exists
os.makedirs(UPLOADS_DIR, exist_ok=True)

# System prompt for answering
# This strict prompt forces the model to cite the exact source and avoid hallucinations.
QA_SYSTEM_PROMPT = """You are a helpful HR assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say "I cannot find that information in the provided HR policy documents".
Always cite the exact source and page numbers from the provided context in your answer. Do not guess or make up facts.

Context: {context}"""

# Prompt for history-aware rephrasing
CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
