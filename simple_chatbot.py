import streamlit as st
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
        return [self.text_chunks[i] for i in indices[0]]

    def get_response(self, user_input):
        """Generate a response from the chatbot."""
        responses = self.retrieve_documents(user_input)
        if responses:
            return " ".join(responses)
        else:
            return "No relevant information found."


# Streamlit application starts here
st.title("Converstional AI Chatbot")

# Initialize the chatbot
chatbot = Chatbot()
chatbot.load_text_chunks('data/text_chunks.txt')
chatbot.create_faiss_index()

# User input
session_title = st.text_input("Enter session title:")
user_input = st.text_area("You:", "")

if st.button("Send"):
    if user_input:
        response = chatbot.get_response(user_input)
        st.write(f"Chatbot: {response}")
    else:
        st.write("Please enter a question.")

# Displaying chat history
if st.button("Show History"):
    st.write("Chat history is not implemented yet, but you can log conversations by enhancing the code.")
