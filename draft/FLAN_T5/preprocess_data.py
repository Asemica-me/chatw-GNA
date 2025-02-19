import os

def preprocess_data(file_path, output_dir, chunk_size=500):
    """
    Preprocesses the content by splitting it into smaller chunks.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into smaller chunks
    chunks = []
    current_chunk = []
    for line in content.split('\n'):
        line = line.strip()
        if line:
            current_chunk.append(line)
            if sum(len(c) for c in current_chunk) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Save chunks into a directory
    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(output_dir, f'chunk_{i}.txt'), 'w', encoding='utf-8') as f:
            f.write(chunk)

    print(f"Processed {len(chunks)} chunks and saved to {output_dir}")

# Example usage
preprocess_data('project_content.txt', 'data_chunks')
