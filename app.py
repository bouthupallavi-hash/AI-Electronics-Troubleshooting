import os
import streamlit as st

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# -----------------------------
# Streamlit page
# -----------------------------

st.set_page_config(
    page_title="AI Electronics Troubleshooting Assistant",
    page_icon="🔧"
)

st.title("🔧 AI-Powered Electronics Troubleshooting Assistant")
st.write("Describe your electronics problem and get troubleshooting guidance.")


# -----------------------------
# Paths
# -----------------------------

DOCUMENTS_FOLDER = "documents"
CHROMA_FOLDER = "chroma_db"


# -----------------------------
# Load documents
# -----------------------------

def load_documents():

    documents = []

    for filename in os.listdir(DOCUMENTS_FOLDER):

        filepath = os.path.join(
            DOCUMENTS_FOLDER,
            filename
        )

        if filename.lower().endswith(".txt"):

            loader = TextLoader(
                filepath,
                encoding="utf-8"
            )

            documents.extend(
                loader.load()
            )

        elif filename.lower().endswith(".pdf"):

            loader = PyPDFLoader(filepath)

            documents.extend(
                loader.load()
            )

    return documents


# -----------------------------
# Create or load vector database
# -----------------------------

@st.cache_resource
def get_vectorstore():

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # If Chroma database already exists,
    # load it instead of creating embeddings again.

    if os.path.exists(CHROMA_FOLDER):

        vectorstore = Chroma(
            persist_directory=CHROMA_FOLDER,
            embedding_function=embeddings,
            collection_name="electronics_troubleshooting"
        )

        return vectorstore


    # First time: load documents

    documents = load_documents()

    if not documents:

        st.error(
            "No TXT or PDF documents found in the documents folder."
        )

        st.stop()


    # Split documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(
        documents
    )


    # Create Chroma database

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER,
        collection_name="electronics_troubleshooting"
    )

    return vectorstore


# -----------------------------
# Load vector database
# -----------------------------

vectorstore = get_vectorstore()


# -----------------------------
# Load AI model
# -----------------------------

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0
)


# -----------------------------
# User question
# -----------------------------

question = st.chat_input(
    "Describe your electronics problem..."
)


if question:

    st.chat_message("user").write(
        question
    )


    # Retrieve only relevant information

    results = vectorstore.similarity_search(
        question,
        k=3
    )


    context = "\n\n".join(
        document.page_content
        for document in results
    )


    # -----------------------------
    # Prompt
    # -----------------------------

    prompt = f"""
You are an AI electronics troubleshooting assistant.

Use only the troubleshooting information provided below.

Troubleshooting information:

{context}

User question:

{question}

Give a simple step-by-step troubleshooting answer.

Do not invent information.

If the problem involves dangerous electrical components,
tell the user to contact a qualified technician.
"""


    # -----------------------------
    # Generate answer
    # -----------------------------

    response = llm.invoke(
        prompt
    )


    st.chat_message("assistant").write(
        response.content
    )