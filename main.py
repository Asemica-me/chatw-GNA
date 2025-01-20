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

# Funzione per caricare il dataset
def load_dataset(dataset_name: str = "gna_kg_dataset.csv") -> pd.DataFrame:
    """Carica un dataset da file CSV e verifica la presenza delle colonne necessarie."""
    data_dir = "./data"
    file_path = os.path.join(data_dir, dataset_name)

    
    try:
        # Verifica se il file esiste prima di caricarlo
        if not os.path.exists(file_path):
            print(f"Errore: Il file {file_path} non esiste.")
            return None
        
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Verifica che le colonne richieste siano presenti
        required_columns = ['body', 'title', 'description', 'url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Attenzione: Colonne mancanti nel dataset: {', '.join(missing_columns)}")

        return df
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        raise

def create_chunks(dataset: pd.DataFrame, chunk_size: int, chunk_overlap: int):
    """Crea chunk informativi dal dataset per l'archiviazione e il recupero."""
    
    # Verifica se la colonna 'body' esiste nel dataset
    if 'body' not in dataset.columns:
        print("Errore: la colonna 'body' non è presente nel dataset.")
        return []

    # Carica e suddividi il dataset
    text_chunks = DataFrameLoader(
        dataset, page_content_column="body"
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
    )

    # Controllo per evitare che il dataset sia vuoto o contenga pochi chunk
    if len(text_chunks) == 0:
        print("Avviso: Nessun chunk è stato generato.")
    
    formatted_chunks = []
    for i, doc in enumerate(text_chunks):
        # Estrai i metadati del chunk
        title = doc.metadata.get("title", "No Title")
        description = doc.metadata.get("description", "No Description")
        content = doc.page_content[:100] + "..."  # Troncamento per evitare output troppo lungo
        url = doc.metadata.get("url", "No URL")

        # Formatta il contenuto del chunk
        final_content = f"TITLE: {title}\nDESCRIPTION: {description}\nBODY: {doc.page_content}\n"
        if 'subtitles' in doc.metadata:
            final_content += f"SUBTITLES: {doc.metadata['subtitles']}\n"
        if 'sections' in doc.metadata:
            final_content += f"SECTIONS: {doc.metadata['sections']}\n"
        if 'plain_text' in doc.metadata:
            final_content += f"PLAIN_TEXT: {doc.metadata['plain_text']}\n"

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
        print("Cartella ./db creata.")

    # Verifica se il vector store esiste
    vector_store_path = "./db/index.faiss"
    if not os.path.exists(vector_store_path):
        print("Vector store non trovato. Creazione di un nuovo vector store...")

        # Log dei chunk
        document_texts = [doc.page_content for doc in chunks if doc.page_content and len(doc.page_content.strip()) > 0]
        if not document_texts:
            raise ValueError("La lista 'document_texts' è vuota o contiene solo testo vuoto.")

        # Debugging: verifica i primi documenti
        print(f"Primi 5 documenti: {document_texts[:5]}")

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
            print(f"Tentativo {attempt + 1} di invocazione API...")
            
            # Invocazione della conversation chain
            response = conversation_chain.invoke({"question": test_message})
            
            # Stampa la risposta ricevuta
            print(f"Risposta ricevuta: {response}")
            return response

        except requests.exceptions.HTTPError as e:
            # Gestione specifica degli errori HTTP
            if e.response and e.response.status_code == 429:
                print("Errore 429: Limite di richieste superato. Attendo prima di riprovare...")
                time.sleep(delay)
            else:
                print(f"Errore HTTP non previsto: {e}")
                raise e  # Rilancia errori non gestiti

        except Exception as e:
            # Gestione generale degli errori
            print(f"Errore durante l'invocazione: {e}")
            if attempt < retries - 1:
                print(f"Riprovo tra {delay} secondi...")
                time.sleep(delay)
            else:
                print("Numero massimo di tentativi raggiunto. Operazione fallita.")
                raise e  # Rilancia l'errore dopo il massimo numero di tentativi

    # Messaggio finale se i tentativi sono stati esauriti
    raise Exception("Numero massimo di tentativi superato senza successo.")

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
                    f"<p style='text-align: left;'><b>AI Assistente</b></p>"
                    f"<p style='text-align: left; {chatbot_style}'><i>{message.content}</i></p>",
                    unsafe_allow_html=True,
                )
    except Exception as e:
        # Logging degli errori
        st.error(f"Si è verificato un errore: {str(e)}")
        print(f"Errore nella gestione della risposta: {e}")  # Debug dell'errore

def main():
    # Carica le variabili di ambiente
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Errore: MISTRAL_API_KEY non trovata. Verifica il file .env.")
        return
    
    # Configura la pagina Streamlit prima di qualsiasi altro comando
    st.set_page_config(
        page_title="GNA Assistente AI",
        page_icon=":bust_in_silhouette:",
    )
    
    # Crea il modello Mistral LLM
    try:
        mistral_llm = create_mistral_llm(api_key)
    except ValueError as e:
        st.error(str(e))
        return
    
    # Carica dataset e chunks
    dataset = load_dataset("gna_kg_dataset.csv")
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
    st.title("Assistente GNA")
    st.subheader("Assistente AI per il progetto GNA")
    st.markdown(
        """
        Questo assistente è stato creato per rispondere a domande sul manuale d'uso dell'applicativo GIS per il progetto Geoportale Nazionale dell’Archeologia (GNA).
        Poni una domanda e l'assistente ti risponderà con la pagina più rilevante della manuale.
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