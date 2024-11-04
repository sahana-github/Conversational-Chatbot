import streamlit as st
from chatbot import chatbot_response, create_faiss_index, load_text_chunks

# Load text chunks and create FAISS index
text_chunks = load_text_chunks('data/text_chunks.txt')
index = create_faiss_index(text_chunks)

st.title("AI Chatbot with Citations")
session_title = st.text_input("Enter session title:", "Chat Session")
chat_history = []

if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You: ")

if st.button("Send"):
    if user_input:
        response = chatbot_response(user_input, index, text_chunks)
        chat_history.append({"User": user_input, "Chatbot": response})
        st.session_state.history.append({"User": user_input, "Chatbot": response})

        for chat in st.session_state.history:
            st.write(f"You: {chat['User']}")
            st.write(f"Chatbot: {chat['Chatbot']}")
    else:
        st.warning("Please enter a question.")
