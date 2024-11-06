# AI-Based Chatbot with Semantic Search and Citations

## Project Overview
This project implements an AI-based chatbot capable of answering user queries using information sourced from a set of PDF documents. The chatbot utilizes **semantic search** techniques to find relevant information, combines retrieval with generative responses through **Retrieval-Augmented Generation (RAG)**, and provides citations for the information it retrieves.

## Key Features
- **Semantic Search**: Understands the meaning behind queries for accurate responses.
- **Document Chunking**: PDF documents are divided into manageable chunks for effective processing.
- **RAG (Retrieval-Augmented Generation)**: Combines document retrieval with generative response capabilities.
- **Citations**: Provides references to the original documents for transparency.
- **Session Management**: Each conversation can be titled, and chat history is maintained.

## Technology Stack
- **Python**: Primary programming language.
- **Transformers Library**: Used for GPT-2 model and tokenizer.
- **Sentence Transformers**: For creating document embeddings.
- **FAISS**: For efficient similarity search and clustering of dense vectors.
- **Streamlit**: Framework for building the user interface.

## Getting Started
### Prerequisites
- Python 3.x
- Required libraries:
  - `transformers`
  - `sentence-transformers`
  - `faiss-cpu`
  - `streamlit`
  
You can install the required libraries using pip:
```bash
pip install transformers sentence-transformers faiss-cpu streamlit
