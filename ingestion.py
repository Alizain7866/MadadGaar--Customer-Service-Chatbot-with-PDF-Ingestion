# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS

# def ingest_pdf(pdf_path):
#     # Load PDF
#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()

#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_documents(documents)

#     # Create embeddings with Ollama
#     embeddings = OllamaEmbeddings(model="llama3.1")  # Ollama embedding model

#     # Store embeddings in FAISS
#     vector_store = FAISS.from_documents(chunks, embeddings)
#     vector_store.save_local("vectors/faiss_index")
#     print(f"PDF ingested and vectorized. Index saved to vectors/faiss_index")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         ingest_pdf(sys.argv[1])


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def ingest_pdf(pdf_path):
    logger.info(f"Starting PDF ingestion for {pdf_path}")
    start_time = time.time()

    # Load PDF
    logger.info("Loading PDF file")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info(f"PDF loaded successfully with {len(documents)} pages")

    # Split text into chunks
    logger.info("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Text split into {len(chunks)} chunks")

    # Create embeddings with Ollama
    logger.info("Initializing Ollama embeddings")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  #
    logger.info("Creating embeddings for document chunks")
    embed_start = time.time()
    vector_store = FAISS.from_documents(chunks, embeddings)
    logger.info(f"Embeddings created in {time.time() - embed_start:.2f} seconds")

    # Save FAISS index
    logger.info("Saving FAISS index to vectors/faiss_index")
    vector_store.save_local("vectors/faiss_index")
    total_time = time.time() - start_time
    logger.info(f"PDF ingestion and vectorization completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ingest_pdf(sys.argv[1])