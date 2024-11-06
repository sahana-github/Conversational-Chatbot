import streamlit as st
from chatbot import Chatbot

# Initialize the chatbot
chatbot = Chatbot()

# Load text chunks and create FAISS index
chatbot.load_text_chunks('data/text_chunks.txt')
chatbot.create_faiss_index()

# Initialize chat history in Streamlit session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Title of the Streamlit app
st.title("AI Chatbot with Citations")

# Session title input
session_title = st.text_input("Enter session title:")

# User input field
user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        # Get chatbot response with citations
        response = chatbot.get_response(user_input)

        # Append both user input and chatbot response to the history
        st.session_state.history.append({"User": user_input, "Chatbot": response})

        # Display chat history
        for chat in st.session_state.history:
            st.write(f"You: {chat['User']}")
            st.write(f"Chatbot: {chat['Chatbot']}")
    else:
        st.warning("Please enter a question.")

# Display chat history when button is clicked
if st.button("Show History"):
    for chat in st.session_state.history:
        st.write(f"You: {chat['User']}")
        st.write(f"Chatbot: {chat['Chatbot']}")
