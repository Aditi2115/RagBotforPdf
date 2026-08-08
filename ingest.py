from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Load PDF
loader = PyPDFLoader(r"C:\Users\ASUS\Downloads\Aditi_Dawange_60Day_Job_Search_Plan.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")


# 2. Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# 3. Local embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 4. Create Chroma vector database
vectorstore = Chroma(
    collection_name="pdf_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# 5. Store chunks
vectorstore.add_documents(chunks)

print("PDF successfully stored in Chroma!")