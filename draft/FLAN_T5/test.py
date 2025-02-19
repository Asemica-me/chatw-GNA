from nltk.translate.bleu_score import sentence_bleu
import os
import faiss
import numpy as np
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

def load_faiss_index(index_path, metadata_path):
    """
    Load FAISS index and associated metadata.
    """
    index = faiss.read_index(index_path)
    file_paths = np.load(metadata_path, allow_pickle=True)
    return index, file_paths

def retrieve_documents(query, index, file_paths, model, top_k=5):
    """
    Retrieve top-k relevant documents using FAISS.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=top_k)  # Retrieve top-5
    retrieved_docs = []
    for idx in indices[0]:
        file_path = file_paths[idx]
        with open(os.path.join('data_chunks', file_path), 'r', encoding='utf-8') as f:
            retrieved_docs.append(f.read())
    return retrieved_docs

def generate_response(query, context, generator):
    """
    Generate a response using the Hugging Face model, combining context efficiently.
    """
    input_text = f"""
    Use the following context to answer the question accurately and in detail:
    Context: {context}

    Question: {query}

    Answer with a detailed explanation, referencing specific information from the context:
    """
    response = generator(input_text, max_length=300, num_return_sequences=1, truncation=True)
    return response[0]['generated_text']

def rank_documents(query, retrieved_docs):
    """
    Rank documents based on relevance using a CrossEncoder.
    """
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict([(query, doc) for doc in retrieved_docs])
    ranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
    return ranked_docs[:3]  # Prendi i 3 documenti più pertinenti

def extract_key_sentences(context, query):
    """
    Extract key sentences from the context based on the query.
    """
    qa_pipeline = pipeline("question-answering", model="deepset/bert-large-uncased-whole-word-masking-squad2")
    results = []
    for doc in context.split("\n"):
        if doc.strip():
            answer = qa_pipeline({'question': query, 'context': doc})
            results.append(answer['answer'])
    return " ".join(results)

# Load FAISS index and metadata
index, file_paths = load_faiss_index('faiss_index.idx', 'file_paths.npy')

# Load models
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline('text2text-generation', model=T5ForConditionalGeneration.from_pretrained('google/flan-t5-base'), tokenizer=T5Tokenizer.from_pretrained('google/flan-t5-base'))

# Input query
query = "What is the purpose of the Geoportale Nazionale dell'Archeologia?"
retrieved_docs = retrieve_documents(query, index, file_paths, retrieval_model)
ranked_docs = rank_documents(query, retrieved_docs)
context = " ".join(ranked_docs)
context = extract_key_sentences(context, query)
response = generate_response(query, context, generator)

# Print the response
print("Generated Response:")
print(response)


############# TEST ##################

reference = [["The purpose of the Geoportale is to provide resources for archaeological data."]]
generated = response.split()

bleu_score = sentence_bleu(reference, generated)
print(f"BLEU Score: {bleu_score}")
