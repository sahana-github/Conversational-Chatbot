import PyPDF2
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer for embedding
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to extract text from PDF with page information
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text_per_page = []
        
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_per_page.append((page_number + 1, page_text))  # (Page number, Text)
    
    return text_per_page

# Function to create embeddings for the extracted text
def create_embeddings(text_per_page):
    sentences = []
    embeddings = []
    
    # Process each page's text
    for page_number, page_text in text_per_page:
        page_sentences = page_text.split("\n")  # Split text into sentences
        for sentence in page_sentences:
            if sentence.strip():  # Only non-empty sentences
                sentences.append((page_number, sentence.strip()))  # (Page, Sentence)
    
    # Create embeddings for each sentence
    sentence_texts = [sentence for _, sentence in sentences]
    embeddings = embedding_model.encode(sentence_texts)
    
    return embeddings, sentences

# Function to create a FAISS index from the embeddings
def create_faiss_index(embeddings):
    dim = len(embeddings[0])  # Length of the embedding vector
    index = faiss.IndexFlatL2(dim)  # Use L2 distance for similarity
    faiss_index = faiss.IndexIDMap2(index)
    faiss_index.add_with_ids(np.array(embeddings), np.arange(len(embeddings)))
    return faiss_index

# Function to save the FAISS index to disk
def save_faiss_index(faiss_index, embeddings, sentences, index_filename='faiss_index.pkl'):
    with open(index_filename, 'wb') as f:
        pickle.dump((faiss_index, embeddings, sentences), f)

# Function to load the FAISS index from disk
def load_precomputed_faiss_index(index_filename='faiss_index.pkl'):
    if os.path.exists(index_filename):
        with open(index_filename, 'rb') as f:
            faiss_index, embeddings, sentences = pickle.load(f)
        return faiss_index, embeddings, sentences
    else:
        return None, None, None

# Function to search the FAISS index for the most similar sentences
def search_faiss_index(faiss_index, query_embedding, top_k=5):  # Reduced to 5 for faster retrieval
    D, I = faiss_index.search(np.array([query_embedding]), top_k)
    return I[0], D[0]

# Combine PDF extraction and embedding generation
def extract_and_embed_pdf(file_path):
    # Extract text from PDF
    text_per_page = extract_text_from_pdf(file_path)
    
    # Create embeddings for the extracted text
    embeddings, sentences = create_embeddings(text_per_page)
    
    # Create a FAISS index for the embeddings
    faiss_index = create_faiss_index(embeddings)
    
    return text_per_page, faiss_index, embeddings, sentences
