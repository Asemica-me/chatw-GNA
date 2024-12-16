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

import streamlit as st

import time
import requests

def load_dataset(dataset_name: str = "dataset_langchain.csv") -> pd.DataFrame:
    """Carica un dataset da file CSV."""
    data_dir = "./data"
    file_path = os.path.join(data_dir, dataset_name)
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset caricato con successo. Numero di righe: {len(df)}")
        print(f"Prime righe del dataset:\n{df.head()}")  # Aggiungi per vedere le prime righe
        return df
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        raise

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

        # Debug: visualizzare il primo chunk
        if i == 0:
            print(f"\nChunk {i + 1}:")
            print(f"Title: {title}")
            print(f"Description: {description}")
            print(f"Content: {content}")
            print(f"URL: {url}")

    return formatted_chunks

def create_or_get_vector_store(chunks: list, api_key: str) -> FAISS:
    """Configura il token HF.
    Crea o carica un vector store basato su FAISS."""
    # Carica le variabili d'ambiente
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Token HuggingFace configurato correttamente.")
    else:
        print("Attenzione: il token HuggingFace non è stato trovato. Potrebbero verificarsi limitazioni.")

    # Inizializza gli embeddings di Mistral
    mistral_embeddings = MistralAIEmbeddings(api_key=api_key)

    # Verifica se il vector store esiste
    if not os.path.exists("./db"):
        os.makedirs("./db")

    if not os.path.exists("./db/faiss.index"):
        print("Vector store non trovato. Creazione di un nuovo vector store...")
        document_texts = [doc.page_content for doc in chunks]
        vector_store = FAISS.from_texts(document_texts, mistral_embeddings)
        vector_store.save_local("./db")
        print("Vector store creato e salvato in ./db.")
    else:
        print("Vector store trovato. Caricamento in corso...")
        vector_store = FAISS.load_local("./db", embeddings=mistral_embeddings)

    num_documents = len(vector_store.docstore._dict)
    print(f"Numero di documenti nel vector store: {num_documents}")

    return vector_store

def create_mistral_llm(api_key: str, model_name: str = "open-mistral-nemo"):
    """Crea un LLM Mistral per la generazione di testo."""
    print(f"Inizializzazione del modello Mistral: {model_name}")
    llm = ChatMistralAI(
        model=model_name,
        temperature=0,
        max_retries=2,
    )
    print("Modello Mistral creato con successo.")
    return llm

def get_conversation_chain(vector_store, api_key: str, model_name: str):
    """
    Ottiene la conversation chain utilizzando Mistral.
    """
    print("Creazione della conversation chain...")
    llm = create_mistral_llm(api_key, model_name)
    
    # Configuriamo la memoria per la conversazione
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Creiamo la catena conversazionale con il recupero
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def invoke_with_retry(conversation_chain, test_message, retries=5, delay=10):
    """
    Funzione per gestire il tentativo e l'errore (con retry).
    """
    for attempt in range(retries):
        try:
            print(f"Invocazione API tentativo {attempt + 1}...")
            response = conversation_chain.invoke({"question": test_message})
            print(f"Risposta ricevuta: {response}")
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Errore 429: Limite di richieste superato. Riprovo...")
                time.sleep(delay)  # Pausa di 5 secondi prima di riprovare
            else:
                print(f"Errore HTTP non previsto: {e}")
                raise e  # Altri errori non gestiti
    print("Numero massimo di tentativi superato.")
    raise Exception("Numero massimo di tentativi superato.")

def handle_style_and_responses(user_question: str, mistral_llm) -> None:
    """
    Handle user input to create the chatbot conversation in Streamlit.

    Args:
        user_question (str): User question
    """
    try:
        # Verifica e inizializzazione della chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Ottieni la risposta dal modello
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response.get("chat_history", [])

        # Definizione degli stili
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
                    f"<p style='text-align: right;'><b>Utente</b></p>"
                    f"<p style='text-align: right; {human_style}'><i>{message.content}</i></p>",
                    unsafe_allow_html=True,
                )
            else:  # Messaggi del chatbot
                st.markdown(
                    f"<p style='text-align: left;'><b>Chatbot</b></p>"
                    f"<p style='text-align: left; {chatbot_style}'><i>{message.content}</i></p>",
                    unsafe_allow_html=True,
                )
    except Exception as e:
        # Logging degli errori
        st.error(f"Si è verificato un errore: {str(e)}")
        print(f"Errore nella gestione della risposta: {e}")  # Debug dell'errore

# if __name__ == "__main__":
#     load_dotenv()

#     api_key = os.getenv("MISTRAL_API_KEY")
#     model_name = os.getenv("MODEL", "open-mistral-nemo")
#     if not api_key:
#         print("Errore: MISTRAL_API_KEY non trovata.")
#         exit()

#     dataset_name = "dataset_langchain.csv"
#     dataset = load_dataset(dataset_name)
#     chunks = create_chunks(dataset, chunk_size=1000, chunk_overlap=100)
#     vector_store = create_or_get_vector_store(chunks, api_key)

#     conversation_chain = get_conversation_chain(vector_store, api_key, model_name)

#     test_message = "Qual è il contenuto principale di questo dataset?"
#     try:
#         response = invoke_with_retry(conversation_chain, test_message)
#         if 'answer' in response:
#             print(f"Risposta dalla conversation chain: {response['answer']}")
#         else:
#             print("Campo 'answer' non trovato nella risposta.")
#     except Exception as e:
#         print(f"Errore durante la conversazione: {e}")

def main(): 
    # Carica le variabili di ambiente
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Errore: MISTRAL_API_KEY non trovata. Verifica il file .env.")
        return

    # Configura la pagina Streamlit prima di qualsiasi altro comando
    st.set_page_config(
        page_title="Documentation Chatbot",
        page_icon=":books:",
    )

    # Crea il modello Mistral LLM
    try:
        mistral_llm = create_mistral_llm(api_key)
        st.write("Modello Mistral LLM configurato correttamente!")
    except ValueError as e:
        st.error(str(e))
        return

    # Carica dataset e chunks
    dataset = load_dataset("dataset_langchain.csv")
    chunks = create_chunks(dataset, chunk_size=1000, chunk_overlap=0)

    # Configura Vector Store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_or_get_vector_store(chunks, api_key)

    # Configura Conversation Chain
    if "conversation" not in st.session_state:
        model_name = "open-mistral-nemo"  # Imposta il nome del modello (es. 'mistral-7b')
        st.session_state.conversation = get_conversation_chain(
            st.session_state.vector_store, api_key, model_name
        )

    # Configura cronologia chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # UI
    st.title("Documentation Chatbot")
    st.subheader("Chatbot per la documentazione del progetto LangChain")
    st.markdown(
        """
        Questo chatbot è stato creato per rispondere a domande sulla documentazione del progetto LangChain.
        Poni una domanda e il chatbot ti risponderà con la pagina più rilevante della documentazione.
        """
    )
    #st.image("https://aunoa.ai/wp-content/uploads/2024/05/tipos-de-chatbots.webp", caption="Chatbot Assistance")
    # st.image("https://images.unsplash.com/photo-1485827404703-89b55fcc595e", caption="Chatbot Assistance") # immagine precedente

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
        '<img src="https://www.cxtoday.com/wp-content/uploads/2019/09/How-Do-Bots-Chatbots-Work.jpg" class="rounded-image" alt="Chatbot Assistance"/>', 
        unsafe_allow_html=True
    )

    # Input utente
    user_question = st.text_input("Cosa vuoi chiedere?")
    if user_question:
        with st.spinner("Elaborando risposta..."):
            handle_style_and_responses(user_question, mistral_llm)

if __name__ == "__main__":
    main()