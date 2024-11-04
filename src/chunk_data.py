def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

if __name__ == "__main__":
    with open('data/extracted_text.txt', 'r',encoding='utf-8') as file:
        extracted_text = file.read()
    
    text_chunks = chunk_text(extracted_text)
    
    # Save chunks for later use (optional)
    with open('data/text_chunks.txt', 'w',encoding='utf-8') as chunk_file:
        for chunk in text_chunks:
            chunk_file.write(chunk + "\n---\n")  # Separate chunks

    print("Text chunking complete!")
