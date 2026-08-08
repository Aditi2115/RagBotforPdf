import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables
load_dotenv()


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="RagBot",
    page_icon="📄"
)

st.title("📄 RagBot")
st.write("Upload a PDF and ask questions about it")


# -----------------------------
# Upload and index the PDF
# -----------------------------

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file is None:
    st.info("Upload a PDF to start chatting.")
    st.stop()


@st.cache_resource(show_spinner="Reading and indexing your PDF...")
def build_vectorstore(pdf_bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_file:
        pdf_file.write(pdf_bytes)
        pdf_path = pdf_file.name

    try:
        documents = PyPDFLoader(pdf_path).load()
    finally:
        os.unlink(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300,
    )
    chunks = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

    return Chroma.from_documents(
        chunks,
        embedding=embeddings,
        collection_name="uploaded_pdf",
    )


try:
    vectorstore = build_vectorstore(uploaded_file.getvalue())
except Exception as error:
    if "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error):
        st.error(
            "Gemini's embedding quota was reached. Wait for the quota window "
            "to reset or use a Gemini API project with billing enabled, then "
            "upload the PDF again."
        )
    else:
        st.error(f"The PDF could not be indexed: {error}")
    st.stop()


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