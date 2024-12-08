import streamlit as st
import os
from typing import Optional, List, Dict, Any

# Explicitly import and configure to avoid schema issues
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain_together import Together

# Configure page
st.set_page_config(page_title="Medical Notes Generator", page_icon="🩺")

def safe_document_loader(file_path: str) -> List[Document]:
    """
    Safely load documents with error handling
    """
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return []

def initialize_system(uploaded_file):
    """Initialize the RAG and note generation system"""
    if not uploaded_file:
        st.warning("Please upload a conversation text file")
        return None, None, None

    

    # Save uploaded file temporarily
    with open("temp_conversation.txt", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the Document
    documents = safe_document_loader("temp_conversation.txt")
    if not documents:
        st.error("Could not load the document")
        return None, None, None

    # Split the Document into Chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create Hugging Face Embeddings
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Create VectorStore with FAISS
    try:
        vectorstore = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, None, None

    # RAG Setup
    retriever = vectorstore.as_retriever()
    
    try:
        llm = Together(
            model="google/gemma-2-27b-it",
            together_api_key="38a11d9280e22f5b8c2e38385f133672f06cd405ca1f2cbfd7216183c451a33e",
            temperature=0.7,
            max_tokens=300
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error setting up QA chain: {e}")
        return None, None, None

    # Read conversation text
    with open("temp_conversation.txt", 'r', encoding='utf-8') as file:
        conversation_text = file.read()

    return qa_chain, conversation_text, retriever

def generate_medical_notes(conversation: str) -> str:
    """Generate SOAP notes from conversation"""

    note_prompt = PromptTemplate(
        input_variables=["conversation"],
        template="""
You are a medical assistant tasked with generating professional medical notes from a patient-doctor conversation.
Here is the conversation:
{conversation}
Generate the medical notes in the SOAP format:
- Subjective: Patient's symptoms and history.
- Objective: Observations, test results, and findings.
- Assessment: Diagnosis and impressions.
- Plan: Treatment recommendations, medications, and follow-ups.

"""
    )
    
    llm = Together(
        model="google/gemma-2-27b-it",
        together_api_key="38a11d9280e22f5b8c2e38385f133672f06cd405ca1f2cbfd7216183c451a33e",
        temperature=0.7,
        max_tokens=300
    )
    chain = LLMChain(llm=llm, prompt=note_prompt)
    
    try:
        notes = chain.run(conversation=conversation)
        return notes
    except Exception as e:
        st.error(f"Error generating medical notes: {e}")
        return "Could not generate medical notes due to an error."

def main():
    st.title("Medical Notes & Conversation Assistant 🩺")

    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
        st.session_state.conversation_text = None
        st.session_state.retriever = None
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # File uploader
    uploaded_file = st.file_uploader("Upload Conversation Text File", type=['txt'])

    # Initialize system button
    if st.button("Initialize System"):
        if uploaded_file:
            try:
                st.session_state.qa_chain, st.session_state.conversation_text, st.session_state.retriever = \
                    initialize_system(uploaded_file)
                if st.session_state.qa_chain:
                    st.success("System Initialized Successfully!")
                else:
                    st.error("Failed to initialize the system")
            except Exception as e:
                st.error(f"Initialization error: {e}")
        else:
            st.error("Please upload a conversation text file")

    # Main interface
    tab1, tab2 = st.tabs(["Question Answering", "SOAP Notes Generation"])

    with tab1:
        st.header("Retrieval-Augmented Question Answering")
        
        if st.session_state.qa_chain:
            # Display chat history
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.chat_message("user").write(message['content'])
                else:
                    st.chat_message("assistant").write(message['content'])
            
            # Chat input
            query = st.chat_input("Ask a question about the conversation")
            
            if query:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    'role': 'user', 
                    'content': query
                })
                
                # Display user message
                st.chat_message("user").write(query)
                
                try:
                    # Get answer using QA chain
                    response = st.session_state.qa_chain(query)
                    answer = response['result']
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant', 
                        'content': answer
                    })
                    
                    # Display assistant response
                    st.chat_message("assistant").write(answer)
                    
                except Exception as e:
                    error_msg = f"Error getting answer: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        'role': 'assistant', 
                        'content': error_msg
                    })
        else:
            st.info("Please initialize the system first")

    with tab2:
        st.header("SOAP Notes Generation")
        
        if st.session_state.conversation_text:
            if st.button("Generate SOAP Notes"):
                try:
                    notes = generate_medical_notes(st.session_state.conversation_text)
                    st.write("**Generated SOAP Notes:**")
                    st.write(notes)
                except Exception as e:
                    st.error(f"Error generating notes: {e}")
        else:
            st.info("Please initialize the system first")

if __name__ == "__main__":
    main()