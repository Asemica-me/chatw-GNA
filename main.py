import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mistralai import Mistral
from langchain_mistralai import ChatMistralAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.rate_limiters import InMemoryRateLimiter
import streamlit as st

import time
import requests

def load_dataset(dataset_name: str = "gna_kg_dataset_new.csv") -> pd.DataFrame:
    """Carica un dataset da file CSV."""
    data_dir = "./data"
    file_path = os.path.join(data_dir, dataset_name)
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset caricato con successo. Numero di righe: {len(df)}")
        #print(f"Prime righe del dataset:\n{df.head()}")
        return df
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        raise

# def create_chunks(dataset: pd.DataFrame, chunk_size: int, chunk_overlap: int):
#     """Crea chunk informativi dal dataset per l'archiviazione e il recupero."""
#     print("Creazione dei chunk in corso...")
#     text_chunks = DataFrameLoader(
#         dataset, page_content_column="body"
#     ).load_and_split(
#         text_splitter=RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
#         )
#     )

#     print(f"Numero di chunk generati: {len(text_chunks)}")

#     formatted_chunks = []
#     for i, doc in enumerate(text_chunks):
#         title = doc.metadata.get("title", "No Title")
#         description = doc.metadata.get("description", "No Description")
#         content = doc.page_content[:100] + "..."  # Troncamento per evitare output troppo lungo
#         url = doc.metadata.get("url", "No URL")

#         final_content = f"TITLE: {title}\nDESCRIPTION: {description}\nBODY: {doc.page_content}\nURL: {url}"
#         doc.page_content = final_content
#         formatted_chunks.append(doc)

#         # Debug: visualizzare il primo chunk
#         if i == 0:
#             print(f"\nChunk {i + 1}:")
#             print(f"Title: {title}")
#             print(f"Description: {description}")
#             print(f"Content: {content}")
#             print(f"URL: {url}")

#         formatted_chunks.append(doc.model_copy(update={"page_content": final_content}))

#     return text_chunks

def create_chunks(dataset: pd.DataFrame, chunk_size: int, chunk_overlap: int):
    """Crea chunk informativi dal dataset per l'archiviazione e il recupero."""
    print("Creazione dei chunk in corso...")
    text_chunks = DataFrameLoader(
        dataset, page_content_column="body"
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
    )

    print(f"Numero di chunk generati: {len(text_chunks)}")

    formatted_chunks = []
    for i, doc in enumerate(text_chunks):
        title = doc.metadata.get("title", "No Title")
        description = doc.metadata.get("description", "No Description")
        content = doc.page_content[:100] + "..."  # Troncamento per evitare output troppo lungo
        url = doc.metadata.get("url", "No URL")

        final_content = f"TITLE: {title}\nDESCRIPTION: {description}\nBODY: {doc.page_content}\nURL: {url}"
        doc.page_content = final_content
        formatted_chunks.append(doc)

        # # Debug per visualizzare il primo chunk
        # if i == 0:
        #     print(f"\nChunk {i + 1}:")
        #     print(f"Title: {title}")
        #     print(f"Description: {description}")
        #     print(f"Content: {content}")
        #     print(f"URL: {url}")

    return formatted_chunks

def create_or_get_vector_store(chunks: list, api_key: str) -> FAISS:
    """Crea o carica un vector store basato su FAISS."""
    # Carica le variabili d'ambiente
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Token HuggingFace configurato correttamente.")
    else:
        print("Attenzione: il token HuggingFace non è stato trovato. Potrebbero verificarsi limitazioni.")

    # Inizializza gli embeddings di Mistral
    mistral_embeddings = MistralAIEmbeddings(api_key=api_key, wait_time=3) #wait_time attesa per la creazione degli embeddings di mistral

    if not os.path.exists("./db"):
        os.makedirs("./db")
        print("Cartella ./db creata.")

    # Verifica se il vector store esiste
    vector_store_path = "./db/index_gna.faiss"
    if not os.path.exists(vector_store_path):
        print("Vector store non trovato. Creazione di un nuovo vector store...")

        # Log dei chunk
        document_texts = [doc.page_content for doc in chunks if doc.page_content and len(doc.page_content.strip()) > 0]
        if not document_texts:
            raise ValueError("La lista 'document_texts' è vuota o contiene solo testo vuoto.")

        # Debugging: verifica i primi documenti
        #print(f"Primi 5 documenti: {document_texts[:5]}")

        # Creazione degli embeddings
        embeddings = mistral_embeddings.embed_documents(document_texts)
        if not embeddings or any(len(embedding) == 0 for embedding in embeddings):
            raise ValueError("Gli embeddings risultano vuoti o malformati.")

        # Creazione del vector store
        vector_store = FAISS.from_texts(document_texts, mistral_embeddings)
        vector_store.save_local("./db")
        print(f"Vector store creato e salvato in {vector_store_path}.")
    else:
        print(f"Vector store trovato in {vector_store_path}. Caricamento in corso...")
        vector_store = FAISS.load_local("./db", embeddings=mistral_embeddings, allow_dangerous_deserialization=True)

    return vector_store

    
def create_mistral_llm(api_key: str, model_name: str = "open-mistral-nemo"):
    """Crea un LLM Mistral per la generazione di testo."""
    print(f"Inizializzazione del modello Mistral: {model_name}")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.1, #means 1 request every 5 seconds (0.2 = 1/2 di secondo)
        check_every_n_seconds=0.1, #wake up every 100 ms 
        max_bucket_size=10 #controls the maximum burst size
    )

    llm = ChatMistralAI(
        model=model_name,
        temperature=0,
        max_retries=2,  # Aumenta i retry
        api_key=api_key,
        rate_limiter=rate_limiter
    )
    print("Modello Mistral creato con successo.")
    return llm

def get_conversation_chain(vector_store, api_key: str, model_name: str, system_message:str, human_message:str):
    """
    Ottiene la conversation chain utilizzando Mistral.
    """
    llm = create_mistral_llm(api_key, model_name)
    
    # Configura la memoria per la conversazione
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Crea la catena conversazionale con il recupero della chat
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        rephrase_question=False,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain

def invoke_with_retry(conversation_chain, question, max_retries=5, initial_delay=1.0, backoff_factor=2.0):
    """Implementa retry con backoff esponenziale"""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = conversation_chain({"question": question})
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Tentativo {attempt+1}: Rate limit raggiunto. Riprovo in {delay} secondi...")
                time.sleep(delay)
                delay *= backoff_factor  # Aumenta il ritardo esponenzialmente
            else:
                raise
    raise Exception("Rate limit: massimo numero di tentativi raggiunto")

def handle_style_and_responses(user_question: str, mistral_llm) -> None:
    """
    Handle user input to create the chatbot conversation in Streamlit.

    Args:
        user_question (str): User question
    """
    if "last_request_time" in st.session_state:
        elapsed = time.time() - st.session_state.last_request_time
        if elapsed < 2.0:  # 2 secondi tra le richieste
            st.warning("Attendi almeno 2 secondi tra una richiesta e l'altra")
            return

    try:


        # Verifica e inizializzazione della chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Ottieni la risposta dal modello
        response = st.session_state.conversation.invoke({"question": user_question})
        st.session_state.chat_history = response.get("chat_history", [])

        # Definizione degli stili di interfaccia web
        human_style = "background-color: #3f444f; border-radius: 10px; padding: 10px;"
        chatbot_style = "border-radius: 10px; padding: 10px;"

        # Rendering dei messaggi
        for i, message in enumerate(st.session_state.chat_history):
            # Controllo struttura del messaggio
            if not hasattr(message, "content") or not message.content:
                st.warning(f"Messaggio non valido alla posizione {i}: {message}")
                continue

            if i % 2 == 0:  # Messaggi dell'utente
                st.markdown(
                    f"<p style='text-align: right;'><b>Utente:</b></p>"
                    f"<p style='text-align: right; {human_style}'><i>{message.content}</i></p>",
                    unsafe_allow_html=True,
                )
            else:  # Messaggi del chatbot
                st.markdown(
                    f"<p style='text-align: left;'><b>Assistente AI:</b></p>"
                    f"<p style='text-align: left; {chatbot_style}'><i>{message.content}</i></p>",
                    unsafe_allow_html=True,
                )
    except Exception as e:
        # Logging degli errori
        st.error(f"Si è verificato un errore: {str(e)}")
        print(f"Errore nella gestione della risposta: {e}")

def main():
    # Carica le variabili di ambiente
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Errore: MISTRAL_API_KEY non trovata. Verifica il file .env.")
        return
    
    # Configura la pagina Streamlit prima di qualsiasi altro comando
    st.set_page_config(
        page_title="Assistente GNA",
        page_icon=":bust_in_silhouette:",
    )
    
    # Crea il modello Mistral LLM
    try:
        mistral_llm = create_mistral_llm(api_key)
    except ValueError as e:
        st.error(str(e))
        return
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """
        You are a helpful chatbot assistant tasked with responding to questions about the WikiMedia user manual of the [Geoportale Nazionale dell’Archeologia (GNA)](https://gna.cultura.gov.it/wiki/index.php/Pagina_principale), managed by Istituto centrale per il catalogo e la documentazione (ICCD).

        You should never answer a question with a question, and you should always respond with the most relevant GNA user manual content.

        Do not answer questions that are not about the project.

        Given a question, you should respond with the most relevant user manual content by following the relevant context below:\n
        {context}
        """
    )

    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
    
    # Carica dataset e chunks
    dataset = load_dataset("gna_kg_dataset.csv")
    chunks = create_chunks(dataset, chunk_size=1000, chunk_overlap=0)
    
    # Configura Vector Store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_or_get_vector_store(chunks, api_key)
    
    # Configura Conversation Chain
    if "conversation" not in st.session_state:
        model_name = "open-mistral-nemo"  
        st.session_state.conversation = get_conversation_chain(
            st.session_state.vector_store, api_key, model_name, system_message_prompt, human_message_prompt
        )

    # Configura cronologia chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # UI
    st.title("Assistente AI per il progetto GNA")
    st.markdown(
        """
        Questo assistente è stato creato per rispondere a domande sul manuale d'uso per il progetto [Geoportale Nazionale dell’Archeologia (GNA)](https://gna.cultura.gov.it/wiki/index.php/Pagina_principale).
        Poni una domanda e l'assistente ti risponderà con il contenuto più rilevante del manuale.
        """
    )

    # Immagine
    st.markdown(
        """
        <style>
        .rounded-image {
            border-radius: 15px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # bordo arrotondato per l'immagine
    st.markdown(
        '<img src="https://github.com/Asemica-me/chatw-GNA/blob/main/data/img.jpg?raw=true" class="rounded-image" alt="Chatbot Assistance"/>', 
        unsafe_allow_html=True
    )

    # Input query utente
    user_question = st.text_input("Cosa vuoi chiedere?")
    if user_question:
        with st.spinner("Elaborando la risposta..."):
            handle_style_and_responses(user_question, mistral_llm)


if __name__ == "__main__":
    main()
