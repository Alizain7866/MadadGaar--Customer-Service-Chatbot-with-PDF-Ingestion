# Customer Service Chatbot

An AI-powered chatbot leveraging PDF ingestion and Retrieval-Augmented Generation (RAG) for customer service automation.

## Setup Instructions
1. **Install Docker** (if using Dockerized environment):
   - Build the Docker image: `docker build -t chatbot .`
   - Run the container: `docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/vectors:/app/vectors chatbot`

2. **Install Dependencies** (if running locally):
   - `pip install -r requirements.txt`

3. **Run the App**:
   - `streamlit run app.py`

4. **Ollama Setup**:
   - Ensure Ollama is installed and running locally or on a server. Update the model names in `ingestion.py` and `retrieval.py` as needed.

## Usage
- Upload a PDF via the Streamlit interface.
- Ask questions based on the PDF content.
- View responses and chat history.

## Directory Structure
- `data/`: Stores uploaded PDFs.
- `vectors/`: Stores FAISS vector index.