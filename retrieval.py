# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate

# def get_response(query):
#     # Load FAISS index and embeddings
#     embeddings = OllamaEmbeddings(model="llama3.1")
#     vector_store = FAISS.load_local("vectors/faiss_index", embeddings, allow_dangerous_deserialization=True)

#     # Initialize Ollama LLM
#     llm = Ollama(model="llama3")  # Replace with desired Ollama model

#     # Define prompt template for RAG
#     prompt_template = """Use the following pieces of context to answer the question. If you don't find a direct match, provide a helpful summary or ask for clarification.
#     Context: {context}
#     Question: {question}
#     Answer:"""
#     PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#     # Set up RetrievalQA chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT}
#     )

#     # Get response
#     result = qa_chain({"query": query})
#     response = result["result"]

#     # Fallback handling
#     if "I don't know" in response or len(response.strip()) < 10:
#         response = "I couldn't find a direct answer. Could you clarify your question, or would you like a summary of the document?"

#     return response

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         print(get_response(sys.argv[1]))


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
import time
from content_moderation import is_safe  # Import the guardrail


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def get_response(query):
    logger.info(f"Processing query: '{query}'")
    start_time = time.time()

    if not is_safe(query):
        logger.warning("Unsafe content detected in query. Aborting response generation.")
        return "Your query appears to contain harmful content and cannot be processed."

    # Load FAISS index and embeddings
    logger.info("Loading Ollama embeddings")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    logger.info("Loading FAISS index from vectors/faiss_index")
    vector_store = FAISS.load_local("vectors/faiss_index", embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS index loaded successfully")

    # Initialize Ollama LLM
    logger.info("Initializing Ollama LLM")
    llm = Ollama(model="mistral:latest")

    # Define prompt template for RAG
    logger.info("Setting up RAG prompt template")
    prompt_template = """Use the following pieces of context to answer the question. If you don't find a direct match, provide a helpful summary or ask for clarification.
    Context: {context}
    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Set up RetrievalQA chain
    logger.info("Configuring RetrievalQA chain")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Get response
    logger.info("Generating response")
    result_start = time.time()
    result = qa_chain({"query": query})
    response = result["result"]
    logger.info(f"Response generated in {time.time() - result_start:.2f} seconds")

    # Fallback handling
    if "I don't know" in response or len(response.strip()) < 10:
        logger.warning("No direct match found, triggering fallback")
        response = "I couldn't find a direct answer. Could you clarify your question, or would you like a summary of the document?"

    total_time = time.time() - start_time
    logger.info(f"Query processing completed in {total_time:.2f} seconds")
    return response

def get_retriever():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vector_store = FAISS.load_local("vectors/faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(get_response(sys.argv[1]))