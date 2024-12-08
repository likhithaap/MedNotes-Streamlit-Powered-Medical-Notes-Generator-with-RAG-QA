# MedNotes: Streamlit-Powered Medical Notes Generator with RAG QA 🩺

A Streamlit application designed to streamline the creation of SOAP-format medical notes and provide Retrieval-Augmented Generation (RAG) question answering from doctor-patient conversations.

## Features
- **SOAP Notes Generation**: Automatically generate professional medical notes in the following format:
  - **Subjective**: Patient's symptoms and history.
  - **Objective**: Observations, test results, and findings.
  - **Assessment**: Diagnosis and impressions.
  - **Plan**: Treatment recommendations, medications, and follow-ups.
- **Question Answering**: Ask context-aware questions about the conversation and get accurate answers using RAG.
- **Advanced NLP Stack**:
  - Document embeddings with **HuggingFace** models.
  - Vector search powered by **FAISS**.
  - Retrieval-Augmented Generation with **LangChain** and **Together LLM**.
- **User-Friendly Interface**: Simple upload, processing, and interaction through Streamlit.

## Getting Started

### Prerequisites
- Python 3.8+
- Install required Python libraries:
  ```bash
  pip install streamlit langchain faiss-cpu huggingface-hub langchain-together
