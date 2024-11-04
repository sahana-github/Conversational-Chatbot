import PyPDF2

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)  # Use PdfReader instead
        for page in reader.pages:  # Iterate through pages
            text += page.extract_text() + "\n"  # Extract text from each page
    return text

if __name__ == "__main__":
    pdf_path = 'C:\\Users\\sahan\\Conversational-Chatbot\\data\\Introduction to Machine Learning with Python.pdf'  # Path to your PDF
    text_data = extract_text_from_pdf(pdf_path)
    
    # Save the extracted text to a file (optional)
    with open('data/extracted_text.txt', 'w',encoding='utf-8') as text_file:
        text_file.write(text_data)
    
    print("Text extraction complete!")
