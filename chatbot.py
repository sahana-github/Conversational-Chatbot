import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Chatbot:

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []

    def load_text_chunks(self, file_path):
        """Load text chunks from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            self.text_chunks = file.read().strip().split('\n---\n')

    def create_faiss_index(self):
        """Generate embeddings and create a FAISS index."""
        embeddings = self.model.encode(self.text_chunks)
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search_index(self, query_vector, k=2):
        """Search the FAISS index for nearest neighbors."""
        query_vector = query_vector.reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        return indices, distances

    def retrieve_documents(self, user_input, k=2):
        """Retrieve relevant documents based on user input."""
        query_vector = self.model.encode([user_input]).astype('float32')
        indices, _ = self.search_index(query_vector, k)
        
        # Retrieve document text and create citations
        responses = [self.text_chunks[i] for i in indices[0]]
        citations = [f"Document {i+1}" for i in indices[0]]  # Citation as Document 1, Document 2, etc.

        return responses, citations

    def get_response(self, user_input):
        """Generate a response from the chatbot, including citations."""
        responses, citations = self.retrieve_documents(user_input)
        
        if responses:
            # Format responses with citations
            response_text = "\n\n".join([f"{response}\n(Citation: {citation})" for response, citation in zip(responses, citations)])
            return response_text
        else:
            return "No relevant information found."
        
    
