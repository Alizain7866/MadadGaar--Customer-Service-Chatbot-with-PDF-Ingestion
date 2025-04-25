# Customer Service Chatbot

An AI-powered chatbot leveraging PDF ingestion and Retrieval-Augmented Generation (RAG) for customer service automation.

2. **Install Dependencies** (if running locally):
   - `pip install -r requirements.txt`

3. **Run the App**:
   - `streamlit run app.py`

4. **Ollama Setup**:
   - Ensure Ollama is installed and running locally or on a server. Update the model names in `ingestion.py` and `retrieval.py` as needed.
   - Models needed are nomic-embed-text and Mistral.
5. **PDFS**:
   - We have curated our own pdfs to test the performance of the Chatbot.
6. Evalutions:
   - You can find the code for evaluations in `evalutions.py`.


## Usage
- Upload a PDF via the Streamlit interface.
- Ask questions based on the PDF content.
- View responses and chat history.
- Ask questions generally related to customer service.
  

## Directory Structure
- `data/`: Stores uploaded PDFs.
- `vectors/`: Stores FAISS vector index.
- `dataset`: Contains the files that are pre-ingested that have been used to test teh chatbot RAG process.
