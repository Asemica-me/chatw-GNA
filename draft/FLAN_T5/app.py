from query_system import retrieve_documents, generate_response, load_faiss_index
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

# Initialize components
index, file_paths = load_faiss_index('faiss_index.idx', 'file_paths.npy')
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline('text2text-generation', model=T5ForConditionalGeneration.from_pretrained('google/flan-t5-base'), tokenizer=T5Tokenizer.from_pretrained('google/flan-t5-base'))

def main():
    print("Welcome to the GNA RAG system!")
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        retrieved_docs = retrieve_documents(query, index, file_paths, retrieval_model)
        context = " ".join(retrieved_docs)
        response = generate_response(query, context, generator)
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main()
