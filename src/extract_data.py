# src/extract_data.py

import PyPDF2

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)  # Use PdfReader instead
        for page in reader.pages:  # Iterate through pages
            text += page.extract_text() + "\n"  # Extract text from each page
    return text

def save_extracted_text(pdf_file, output_file):
    """Extract text from PDF and save it to a text file."""
    text = extract_text_from_pdf(pdf_file)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Text extraction complete! Saved to {output_file}")

if __name__ == "__main__":
    # Path to your PDF file
    pdf_path = 'data/Introduction to Machine Learning with Python.pdf'
    output_txt_file = 'data/extracted_text.txt'
    save_extracted_text(pdf_path, output_txt_file)
