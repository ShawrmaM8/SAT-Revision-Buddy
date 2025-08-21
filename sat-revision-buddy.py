
#SAT revision buddy

import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone, ServerlessSpec

# Load API keys from Streamlit secrets
hf_api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone
index_name = "sat-notes"

pc = Pinecone(api_key=pinecone_api_key)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# App title
st.title("SAT Revision Buddy")

# Sidebar for file upload and processing
with st.sidebar:
    st.header("Upload Your SAT Notes")
    uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])
    
    if uploaded_file and st.button("Process Notes"):
        with st.spinner("Processing notes (may take a minute)..."):
            try:
                # Save uploaded file temporarily
                temp_file = f"temp_{uploaded_file.name}"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Load document
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_file)
                else:
                    loader = Docx2txtLoader(temp_file)
                documents = loader.load()
                
                # Clean up temp file
                os.remove(temp_file)
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)
                
                # Embed and store in Pinecone
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vectorstore = Pinecone.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    index_name=index_name
                )
                st.success("Notes processed! Ready to chat, summarize, or quiz.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Main chat interface
if st.session_state.vectorstore:
    # Initialize LLM
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_api_key,
        temperature=0.7
    )
    
    # Contextualize query for history-aware retrieval
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question that can be understood without the chat history. "
        "Do NOT answer, just reformulate if needed."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}), contextualize_q_prompt
    )
    
    # System prompt for SAT assistant
    qa_system_prompt = (
        "You are an SAT revision assistant. Use the provided context to answer questions, "
        "summarize notes, or generate SAT-style quizzes. Keep answers concise and accurate. "
        "For quizzes, provide 3 questions (multiple-choice where appropriate) with answers. "
        "If you don't know the answer, say so. Context:\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Chat input
    user_input = st.chat_input("Ask a question, request a summary, or type 'quiz me'...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Handle quiz mode
        if "quiz" in user_input.lower():
            quiz_prompt = (
                "Generate a SAT-style quiz with 3 questions based on the provided notes. "
                "Include multiple-choice options where appropriate, and provide answers at the end."
            )
            response = rag_chain.invoke({"input": quiz_prompt, "chat_history": st.session_state.memory.chat_memory.messages})
        else:
            response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.memory.chat_memory.messages})
        
        # Update memory and history
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(response["answer"])
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
else:

    st.info("Upload and process your SAT notes in the sidebar to start.")

