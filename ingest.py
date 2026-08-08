from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Load PDF
loader = PyPDFLoader(r"C:\Users\ASUS\Downloads\Aditi_Dawange_60Day_Job_Search_Plan.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")


# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)


# Chroma
vectorstore = Chroma(
    collection_name="pdf_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# Store documents
vectorstore.add_documents(chunks)

print("PDF successfully stored in Chroma!")