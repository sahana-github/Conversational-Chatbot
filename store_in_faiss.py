import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load your model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_text_chunks(file_path):
    """Load text chunks from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text_chunks = file.read().strip().split('\n---\n')
    return text_chunks

def create_faiss_index(text_chunks):
    """Generate embeddings and create a FAISS index."""
    embeddings = model.encode(text_chunks)
    embeddings = np.array(embeddings).astype('float32')  # Convert to float32

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index

def retrieve_documents(user_input, index, text_chunks, k=2):
    """Retrieve relevant documents based on user input."""
    query_vector = model.encode([user_input]).astype('float32')
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    
    responses = [text_chunks[i] for i in indices[0]]
    citations = [f"Document {i+1}" for i in indices[0]]  # Simple citation; modify as needed

    return responses, citations
