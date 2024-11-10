# src/chunk_data.py

def chunk_text(text, chunk_size=512):
    """Chunk the text into smaller pieces of the specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def save_chunks_to_file(chunks, output_file):
    """Save the chunks to a text file, separated by a delimiter."""
    with open(output_file, 'w', encoding='utf-8') as file:
        for chunk in chunks:
            file.write(chunk + "\n---\n")  # Separate chunks by delimiter
    print(f"Text chunking complete! Saved to {output_file}")

if __name__ == "__main__":
    # Read extracted text from file
    with open('data/extracted_text.txt', 'r', encoding='utf-8') as file:
        extracted_text = file.read()

    # Chunk the text into smaller pieces
    text_chunks = chunk_text(extracted_text, chunk_size=512)

    # Save chunks to a new text file
    output_chunks_file = 'data/text_chunks.txt'
    save_chunks_to_file(text_chunks, output_chunks_file)
