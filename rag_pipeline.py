from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from vector_store import get_retriever
from config import QA_SYSTEM_PROMPT, CONTEXTUALIZE_Q_SYSTEM_PROMPT

def get_llm():
    """
    Initializes the LLM. 
    Note: Requires GROQ_API_KEY environment variable. 
    """
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0)

def build_rag_chain():
    """
    Builds the LCEL History-Aware Conversational RAG pipeline.
    
    Architectural decision: A History-Aware Retriever is used because in a conversation, 
    follow-up queries often refer back to previous messages (e.g., "What about the second one?"). 
    This retriever uses the LLM to rewrite the query into a standalone question before vector search.
    """
    llm = get_llm()
    retriever = get_retriever()

    # 1. History-aware retriever setup
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Question Answering setup
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Final RAG Chain combining retrieval and QA
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def get_answer(query: str, chat_history: list):
    """
    Processes the user query and returns the answer and source documents.
    chat_history should be a list of BaseMessage (HumanMessage, AIMessage).
    """
    rag_chain = build_rag_chain()
    
    response = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    return response["answer"], response["context"]
