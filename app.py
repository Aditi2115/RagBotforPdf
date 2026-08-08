import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄"
)

st.title("📄 PDF Chatbot")
st.write("Ask questions about your PDF")


# -----------------------------
# Local embedding model
# -----------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -----------------------------
# Load Chroma database
# -----------------------------

vectorstore = Chroma(
    collection_name="pdf_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# -----------------------------
# Retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# -----------------------------
# Gemini LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# -----------------------------
# Chat input
# -----------------------------

question = st.chat_input(
    "Ask something about the PDF..."
)


# -----------------------------
# Process question
# -----------------------------

if question:

    # Display user question
    st.chat_message("user").write(question)

    # Retrieve relevant chunks
    documents = retriever.invoke(question)

    # Combine retrieved chunks
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create prompt
    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

Do not use your own knowledge.

If the answer cannot be found in the context,
say exactly:

"I couldn't find that information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    # Ask Gemini
    response = llm.invoke(prompt)

    # Gemini may return structured content
    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    # Display answer
    st.chat_message("assistant").write(answer)