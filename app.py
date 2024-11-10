import streamlit as st
import os
from helpers import extract_and_embed_pdf, save_faiss_index, load_precomputed_faiss_index, search_faiss_index
from model import generate_answer_from_context, generate_answer
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer for embedding
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Preload the FAISS index if it exists
if 'faiss_index' not in st.session_state:
    faiss_index, embeddings, sentences = load_precomputed_faiss_index()
    if faiss_index is not None:
        st.session_state['faiss_index'] = faiss_index
        st.session_state['embeddings'] = embeddings
        st.session_state['sentences'] = sentences
    else:
        st.session_state['faiss_index'] = None
        st.session_state['embeddings'] = None
        st.session_state['sentences'] = None

if 'pdf_text' not in st.session_state:
    st.session_state['pdf_text'] = None

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'citations' not in st.session_state:
    st.session_state['citations'] = []

# Title of the app
st.title("Question Answering System")

# Sidebar for session info
st.sidebar.header("Session Info")
session_title = f"Session ({len(st.session_state['history'])} messages)"
st.sidebar.subheader(session_title)

# Display conversation history in sidebar
st.sidebar.subheader("Conversation History:")
for entry in st.session_state['history']:
    st.sidebar.write(f"Q: {entry['question']}")
    st.sidebar.write(f"A: {entry['answer'][:100]}...")  # Show first 100 chars of answer

# Display citations in sidebar
st.sidebar.subheader("Citations:")
for citation in st.session_state['citations']:
    st.sidebar.write(f"Page {citation['page']}: {citation['sentence']}")

# Mode selection for normal QA or PDF-based QA
mode = st.radio("Select Mode", ("Normal QA", "PDF-based QA"))

if mode == "Normal QA":
    # User input for normal QA
    question = st.text_input("Ask a question")

    if question:
        # Generate answer using a smaller, faster model like Flan-T5-Base
        answer = generate_answer(question)

        # Display the answer
        st.subheader("Answer")
        st.write(answer)

        # Store question and answer in session history
        st.session_state['history'].append({"question": question, "answer": answer})

elif mode == "PDF-based QA":
    # File uploader for PDFs
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        file_path = os.path.join('assets', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Extract text and generate embeddings from the PDF (cache this)
        pdf_text, faiss_index, embeddings, sentences = extract_and_embed_pdf(file_path)
        st.session_state['pdf_text'] = pdf_text
        st.session_state['faiss_index'] = faiss_index
        st.session_state['embeddings'] = embeddings
        st.session_state['sentences'] = sentences

        # Save the FAISS index and embeddings to disk for future use
        save_faiss_index(faiss_index, embeddings, sentences)

        st.success("PDF uploaded and indexed successfully!")

    # Text input for PDF-based QA
    question = st.text_input("Ask a question")

    if question and st.session_state['faiss_index'] is not None:
        # Get embedding for the question and search the FAISS index
        question_embedding = embedding_model.encode([question])[0]
        indices, distances = search_faiss_index(st.session_state['faiss_index'], question_embedding, top_k=5)

        # Get the relevant context for answering the question (retrieve top 5 sentences)
        context = "\n".join([st.session_state['sentences'][i][1] for i in indices[:5]])

        # Generate the answer based on context
        answer = generate_answer_from_context(question, context)

        # Display the answer
        st.subheader("Answer")
        st.write(answer)

        # Save citations (sentences with page numbers) in session state
        citations = [{"page": st.session_state['sentences'][i][0], "sentence": st.session_state['sentences'][i][1]} for i in indices[:5]]
        st.session_state['citations'] = citations

        # Store question and answer in session history
        st.session_state['history'].append({"question": question, "answer": answer})
