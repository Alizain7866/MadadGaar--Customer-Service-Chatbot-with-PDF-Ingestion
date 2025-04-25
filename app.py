# import streamlit as st
# from ingestion import ingest_pdf
# from retrieval import get_response

# # Streamlit app configuration
# st.set_page_config(page_title="Customer Service Chatbot", layout="wide")

# # Session state for chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Title and description
# st.title("AI-Powered Customer Service Chatbot")
# st.write("Upload a PDF and ask questions based on its content!")

# # PDF upload section
# uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
# if uploaded_file:
#     with open("data/uploaded_file.pdf", "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     st.success("PDF uploaded successfully!")
#     ingest_pdf("data/uploaded_file.pdf")  # Process the uploaded PDF
#     st.write("PDF processed and ready for queries.")

# # Chat interface
# st.subheader("Chat with the Bot")
# user_query = st.text_input("Enter your question:")
# if user_query:
#     with st.spinner("Generating response..."):
#         response = get_response(user_query)
#         st.session_state.messages.append({"role": "user", "content": user_query})
#         st.session_state.messages.append({"role": "bot", "content": response})

# # Display chat history
# for message in st.session_state.messages:
#     if message["role"] == "user":
#         st.write(f"**You:** {message['content']}")
#     else:
#         st.write(f"**Bot:** {message['content']}")

# if __name__ == "__main__":
#     st.write("Running on http://localhost:8501")


import streamlit as st
from ingestion import ingest_pdf
from retrieval import get_response
import logging
import time
from content_moderation import is_safe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="Customer Service Chatbot", layout="wide")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("AI-Powered Customer Service Chatbot")
st.write("Upload a PDF and ask questions based on its content!")

# PDF upload section
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_path = "data/uploaded_file.pdf"
    logger.info(f"Starting PDF upload process for {uploaded_file.name}")
    with open(pdf_path, "wb") as f:
        logger.info(f"Writing uploaded file to {pdf_path}")
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully!")
    logger.info("Calling ingest_pdf function")
    start_time = time.time()
    ingest_pdf(pdf_path)
    logger.info(f"PDF ingestion completed in {time.time() - start_time:.2f} seconds")
    st.write("PDF processed and ready for queries.")

# Chat interface
st.subheader("Chat with the Bot")
user_query = st.text_input("Enter your question:")
if user_query:
    logger.info(f"Received user query: '{user_query}'")
    # if not is_safe(user_query):
    #     st.error("⚠️ Your question contains unsafe or harmful content and cannot be processed.")
    #     logger.warning("Blocked unsafe query due to content safety violation.")
    #     st.session_state.messages.append({"role": "user", "content": user_query})
    #     st.session_state.messages.append({"role": "bot", "content": "⚠️ I'm sorry, but I can't process that request due to safety concerns."})
    # else:
    with st.spinner("Generating response..."):
        logger.info("Calling get_response function")
        start_time = time.time()
        response = get_response(user_query)
        logger.info(f"Response generated in {time.time() - start_time:.2f} seconds")
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "bot", "content": response})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Bot:** {message['content']}")

if __name__ == "__main__":
    logger.info("Starting Streamlit app on http://localhost:8501")
    st.write("Running on http://localhost:8501")