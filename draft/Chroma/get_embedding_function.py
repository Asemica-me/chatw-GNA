from langchain_openai import OpenAIEmbeddings

def get_embedding_function():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="sk-ijklmnopabcd5678ijklmnopabcd5678ijklmnop")
    return embeddings
