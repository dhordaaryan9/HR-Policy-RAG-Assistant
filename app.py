import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from document_loader import load_and_split_document
from vector_store import create_vector_store
from rag_pipeline import get_answer
from config import UPLOADS_DIR

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

st.set_page_config(page_title="HR Policy Assistant", page_icon="📝", layout="centered")

st.title("📝 HR Policy Chatbot")
st.markdown("Ask any questions about the HR policies!")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for document ingestion
with st.sidebar:
    st.header("Document Upload")
    
    uploaded_file = st.file_uploader("Upload HR Policy (PDF)", type="pdf")
    
    if st.button("Index Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                # Save uploaded file
                file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load, split, and index
                splits = load_and_split_document(file_path)
                create_vector_store(splits)
                st.success(f"Successfully indexed {uploaded_file.name}!")
        else:
            st.warning("Please upload a PDF first.")

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for idx, doc in enumerate(msg["sources"]):
                    source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", "Unknown")
                    st.markdown(f"**Source {idx+1}:** {source_name} (Page {page})")
                    st.text(doc.page_content[:200] + "...")

# Accept user input
if prompt := st.chat_input("Ask a question about HR policies..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = get_answer(prompt, st.session_state.chat_history)
                st.markdown(answer)
                
                # Display sources
                if sources:
                    with st.expander("Sources"):
                        for idx, doc in enumerate(sources):
                            source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                            page = doc.metadata.get("page", "Unknown")
                            st.markdown(f"**Source {idx+1}:** {source_name} (Page {page})")
                            st.text(doc.page_content[:200] + "...")
                
                # Append to session state
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as e:
                st.error(f"Error processing request: {str(e)}\n\n(Did you set your GROQ_API_KEY?)")
