import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import streamlit as st

from transformers import AutoTokenizer, AutoModel
import torch
import faiss

def load_dataset(dataset_name:str="dataset.csv") -> pd.DataFrame:
    """
    Load dataset from file_path

    Args:
        dataset_name (str, optional): Dataset name. Defaults to "dataset.csv".

    Returns:
        pd.DataFrame: Dataset
    """
    data_dir = "./data"
    file_path = os.path.join(data_dir, dataset_name)
    df = pd.read_csv(file_path)
    return df

# Create chunks with sliding window
def create_chunks(dataset: pd.DataFrame, chunk_size: int, chunk_overlap: int):
    """
    Create informational chunks from the dataset

    Args:
        dataset (pd.DataFrame): Dataset Pandas
        chunk_size (int): How many informational chunks?
        chunk_overlap (int): How many overlapping chunks?

    Returns:
        list: List of Document objects (chunks)
    """
    text_chunks = DataFrameLoader(
        dataset, page_content_column="body"
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
    )

    # Ensure that the chunks are in the correct format (Document objects)
    formatted_chunks = []
    for doc in text_chunks:
        title = doc.metadata.get("title", "No Title")
        description = doc.metadata.get("description", "No Description")
        content = doc.page_content
        url = doc.metadata.get("url", "No URL")

        # Construct the content with proper structure
        final_content = f"TITLE: {title}\nDESCRIPTION: {description}\nBODY: {content}\nURL: {url}"
        
        # Reassign the formatted content back to the page_content
        doc.page_content = final_content

        # Append the properly formatted Document to the list
        formatted_chunks.append(doc)

    return formatted_chunks

def create_hf_embeddings():
    """
    Create HuggingFaceEmbeddings using a pre-trained model.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_or_get_vector_store(chunks: list) -> FAISS:
    """
    Create or get vector store.

    Args:
        chunks (list): List of chunks.

    Returns:
        FAISS: Vector store.
    """
    # Load HuggingFace embeddings
    hf_embeddings = create_hf_embeddings()

    # Ensure the 'db' directory exists
    if not os.path.exists("./db"):
        os.makedirs("./db")

    # Check if the FAISS index already exists
    if not os.path.exists("./db/faiss.index"):
        print("CREATING DB")
        document_texts = [doc.page_content for doc in chunks]
        vector_store = FAISS.from_texts(texts=document_texts, embedding=hf_embeddings)
        vector_store.save_local("./db")
    else:
        print("LOADING DB")
        vector_store = FAISS.load_local(
            "./db",
            embeddings=hf_embeddings,
            allow_dangerous_deserialization=True  # Explicitly allow deserialization
        )

    return vector_store


def get_conversation_chain(vector_store: FAISS, system_message: str, human_message: str) -> ConversationalRetrievalChain:
    """
    Get the chatbot conversation chain using a text generation model.

    Args:
        vector_store (FAISS): Vector store (FAISS index)
        system_message (str): System message
        human_message (str): Human message

    Returns:
        ConversationalRetrievalChain: Chatbot conversation chain
    """
    llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/distilgpt2",
    model_kwargs={"huggingface_api_token": os.getenv("HUGGINGFACEHUB_API_TOKEN")}
        )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),  # No change needed
        memory=memory,
    )
    return conversation_chain


def handle_style_and_responses(user_question: str) -> None:
    """
    Handle user input to create the chatbot conversation in Streamlit

    Args:
        user_question (str): User question
    """
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    human_style = "background-color: #e6f7ff; border-radius: 10px; padding: 10px;"
    chatbot_style = "background-color: #f9f9f9; border-radius: 10px; padding: 10px;"

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(
                f"<p style='text-align: right;'><b>Utente</b></p> <p style='text-align: right;{human_style}'> <i>{message.content}</i> </p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='text-align: left;'><b>Chatbot</b></p> <p style='text-align: left;{chatbot_style}'> <i>{message.content}</i> </p>",
                unsafe_allow_html=True,
            )

def main():
    load_dotenv()  # Load environment variables
    dataset = load_dataset("dataset.csv")  # Load dataset
    chunks = create_chunks(dataset, chunk_size=1000, chunk_overlap=0)  # Create chunks

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_or_get_vector_store(chunks)

    if "conversation" not in st.session_state:
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            """
            You are a chatbot tasked with responding to questions about the documentation of the LangChain library and project.
            Always respond with the most relevant documentation page.
            Do not answer questions unrelated to LangChain.
            Context:
            {context}
            """
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
        st.session_state.conversation = get_conversation_chain(
            st.session_state.vector_store, system_message_prompt, human_message_prompt
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.set_page_config(
        page_title="Documentation Chatbot",
        page_icon=":books:",
    )

    st.title("Documentation Chatbot")
    st.subheader("Chatbot per la documentazione del progetto LangChain")
    st.markdown(
        """
        Questo chatbot è stato creato per rispondere a domande sulla documentazione del progetto LangChain.
        Poni una domanda e il chatbot ti risponderà con la pagina più rilevante della documentazione.
        """
    )
    st.image("https://images.unsplash.com/photo-1485827404703-89b55fcc595e", caption="Chatbot Assistance") 

    user_question = st.text_input("Cosa vuoi chiedere?")
    if user_question:
        with st.spinner("Elaborando risposta..."):
            handle_style_and_responses(user_question)

if __name__ == "__main__":
    main()
