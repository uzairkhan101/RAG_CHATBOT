# RAG Chatbot

AI-powered chatbot that lets you upload documents and 
ask questions about them using Retrieval-Augmented Generation.

## Technologies Used
- Python, FastAPI, LlamaIndex, ChromaDB
- HuggingFace Embeddings, TinyLlama
- Streamlit, REST APIs

## How to Run
1. Clone the repository
   git clone https://github.com/yourusername/rag-chatbot.git

2. Install dependencies
   pip install -r requirements.txt

3. Start the backend
   uvicorn main:app --reload

4. Start the frontend
   streamlit run app.py

## Features
- Upload PDF or TXT documents
- Ask questions in natural language
- RAG pipeline with vector search
- Named Entity Recognition (NER)
- Sentiment Analysis

💡 How to Use
Select the action from the sidebar:
🔧 Upload Document
♻️ Reset previous collection (optional)

Start the indexing and Q&A by clicking **Upload**, then type your question in the chat box.
