import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def create_faiss_index(data_dir, index_path):
    """
    Create a FAISS index from text chunks.
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed text chunks
    embeddings = []
    file_paths = []
    for file_name in os.listdir(data_dir):
        with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
            content = f.read()
            embedding = model.encode(content)
            embeddings.append(embedding)
            file_paths.append(file_name)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)

    # Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index and metadata
    faiss.write_index(index, index_path)
    np.save('file_paths.npy', file_paths)
    print(f"Index created and saved at {index_path}")

# Example usage
create_faiss_index('data_chunks', 'faiss_index.idx')
