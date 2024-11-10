from transformers import pipeline

# Initialize the question-answering model (using the smaller Flan-T5-Base model for faster performance)
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")  # Switched to Flan-T5-Base

# Function to generate an answer for normal QA (non-PDF based)
def generate_answer(question):
    # Add detailed prompt to encourage a comprehensive answer
    input_text = f"Provide a detailed, informative, and complete answer to the following question: {question}"
    result = qa_model(input_text)
    return result[0]['generated_text']

# Function to generate an answer based on PDF content (RAG system)
def generate_answer_from_context(question, context):
    # Construct a detailed prompt that incorporates both the question and the context
    input_text = f"Question: {question} Context: {context}. Provide a detailed and thorough answer based on this context."
    result = qa_model(input_text)
    return result[0]['generated_text']
