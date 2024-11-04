import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from store_in_faiss import retrieve_documents, create_faiss_index
from utils import log_chat_history  # Make sure this function exists

# Load GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set padding token if not defined
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

def load_text_chunks(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_chunks = file.read().strip().split('\n---\n')
    return text_chunks

def generate_response(input_text):
    inputs = gpt2_tokenizer.encode(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = gpt2_model.generate(inputs, max_length=150, pad_token_id=gpt2_tokenizer.eos_token_id)
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def chatbot_response(user_input, index, text_chunks):
    responses = retrieve_documents(user_input, index, text_chunks)

    # Flatten the list of responses if necessary
    if isinstance(responses[0], list):
        # If the first response is a list, flatten the entire list
        context = " ".join(item for sublist in responses for item in sublist)
    else:
        context = " ".join(responses) if responses else "No relevant information found."

    return generate_response(context)


if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' to end the session.")
    session_title = input("Enter session title: ")
    chat_history = []

    text_chunks = load_text_chunks('data/text_chunks.txt')
    index, embeddings = create_faiss_index(text_chunks)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = chatbot_response(user_input, index, text_chunks)
        chat_history.append({"User": user_input, "Chatbot": response})
        print(f"Chatbot: {response}")

    log_chat_history(session_title, chat_history)
    print("Chat history logged.")
